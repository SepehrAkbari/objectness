package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"image"
	"image/jpeg"
	_ "image/png"
	"math/rand"
	"time"
	"runtime"
)

const totalCropsPerImage = 20
const numLowSaliencyCrops = 5
const lowSaliencyCropTargetWidth = 224
const lowSaliencyCropTargetHeight = 224

var (
	dataDir = "../data/testing"
	bingProcessorDir = "./bing_processor"
	frcnnProcessorDir = "./frcnn_processor"
	bingExecutablePath = filepath.Join(bingProcessorDir, "build", "BingCropperSingle")
	frcnnScriptPath = filepath.Join(frcnnProcessorDir, "src", "rp_rcnn_single.py")
	finalOutputDir = "./output"
	finalCropsDir = filepath.Join(finalOutputDir, "crops")
	finalCSVFile = filepath.Join(finalOutputDir, "combined_data.csv")
	tempProcessingBaseDir = "./temp_processing"
	bingExecutablePath string
)

func init() {
	if runtime.GOOS == "windows" {
		bingExecutablePath = filepath.Join(bingProcessorDir, "build", "Release", "BingCropperSingle.exe")
	} else {
		bingExecutablePath = filepath.Join(bingProcessorDir, "build", "BingCropperSingle")
	}
}

type CropMeta struct {
	RelativeCropPath string
	X int
	Y int
	Width int
	Height int
	Score float64
}

func getImageDimensions(imagePath string) (int, int, error) {
	file, err := os.Open(imagePath)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to open image %s: %w", imagePath, err)
	}
	defer file.Close()

	config, _, err := image.DecodeConfig(file)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to decode image config %s: %w", imagePath, err)
	}
	return config.Width, config.Height, nil
}

func generateAndSaveLowSaliencyCrop(
	originalImagePath string,
	originalFilename string,
	cropIdx int,
	isWrongFile string,
	cropX, cropY, cropWidth, cropHeight int,
	csvWriter *csv.Writer) error {

	finalCropFilename := fmt.Sprintf("%s_lowsaliency_crop%d.jpg", strings.TrimSuffix(originalFilename, filepath.Ext(originalFilename)), cropIdx)
	destCropPath := filepath.Join(finalCropsDir, finalCropFilename)

	srcFile, err := os.Open(originalImagePath)
	if err != nil {
		fmt.Printf("  [Error] Opening original image %s for low-saliency crop: %v\n", originalImagePath, err)
		return err
	}
	defer srcFile.Close()

	img, _, err := image.Decode(srcFile)
	if err != nil {
		fmt.Printf("  [Error] Decoding image %s for low-saliency crop: %v\n", originalImagePath, err)
		return err
	}

	cropRect := image.Rect(cropX, cropY, cropX+cropWidth, cropY+cropHeight)

	type subImager interface {
		SubImage(r image.Rectangle) image.Image
	}
	subImg, ok := img.(subImager)
	if !ok {
		fmt.Printf("  [Error] Image type does not support SubImage for %s\n", originalImagePath)
		return fmt.Errorf("image type does not support SubImage")
	}
	croppedImage := subImg.SubImage(cropRect)

	destFile, err := os.Create(destCropPath)
	if err != nil {
		fmt.Printf("  [Error] Creating dest file %s for low-saliency crop: %v\n", destCropPath, err)
		return err
	}
	defer destFile.Close()

	err = jpeg.Encode(destFile, croppedImage, &jpeg.Options{Quality: 90})
	if err != nil {
		fmt.Printf("  [Error] Encoding/saving low-saliency crop %s: %v\n", destCropPath, err)
		return err
	}

	x1, y1 := cropX, cropY
	x2, y2 := cropX+cropWidth, cropY+cropHeight

	csvRecord := []string{
		originalFilename,
		strconv.Itoa(cropIdx),
		strconv.Itoa(x1), strconv.Itoa(y1),
		strconv.Itoa(x2), strconv.Itoa(y1),
		strconv.Itoa(x1), strconv.Itoa(y2),
		strconv.Itoa(x2), strconv.Itoa(y2),
		isWrongFile,
		"FALSE",
		"FALSE",
	}
	if err := csvWriter.Write(csvRecord); err != nil {
		fmt.Printf("  [Error] Writing record to final CSV for low-saliency crop %s: %v\n", originalFilename, err)
		return err
	}
	return nil
}

func main() {
	rand.Seed(time.Now().UnixNano())

	if err := os.MkdirAll(finalCropsDir, 0755); err != nil {
		fmt.Printf("Error creating final crops directory: %v\n", err)
		os.Exit(1)
	}
	if err := os.MkdirAll(tempProcessingBaseDir, 0755); err != nil {
		fmt.Printf("Error creating temp base directory: %v\n", err)
		os.Exit(1)
	}

	csvFile, err := os.Create(finalCSVFile)
	if err != nil {
		fmt.Printf("Error creating final CSV file: %v\n", err)
		os.Exit(1)
	}
	defer csvFile.Close()
	csvWriter := csv.NewWriter(csvFile)
	header := []string{"original_filename", "crop_idx", "top_left_x", "top_left_y", "top_right_x", "top_right_y", "bottom_left_x", "bottom_left_y", "bottom_right_x", "bottom_right_y", "WRONG_file", "FRCNN_source", "BING_source"}
	if err := csvWriter.Write(header); err != nil {
		fmt.Printf("Error writing CSV header: %v\n", err)
		os.Exit(1)
	}
	csvWriter.Flush()

	entries, err := os.ReadDir(dataDir)
	if err != nil {
		fmt.Printf("Error reading data directory: %v\n", err)
		os.Exit(1)
	}

	var validImages []os.DirEntry
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		filename := strings.ToLower(entry.Name())
		if strings.HasSuffix(filename, ".jpg") || strings.HasSuffix(filename, ".jpeg") || strings.HasSuffix(filename, ".png") {
			validImages = append(validImages, entry)
		}
	}
	totalImages := len(validImages)

	fmt.Printf("\nProcessing images from %s\n", dataDir)

	for idx, entry := range validImages {
		imageFilename := entry.Name()
		imagePath := filepath.Join(dataDir, imageFilename)
		imageBaseFilename := strings.TrimSuffix(imageFilename, filepath.Ext(imageFilename))

		fmt.Printf("\nProcessing %s (%d/%d)\n", imageFilename, idx+1, totalImages)

		currentTempDir := filepath.Join(tempProcessingBaseDir, imageBaseFilename+"_temp")
		if err := os.MkdirAll(filepath.Join(currentTempDir, "crops"), 0755); err != nil {
			fmt.Printf("  [Error] Creating temp dir for %s: %v. Skipping.\n", imageFilename, err)
			continue
		}

		isWrongFile := "FALSE"
		if strings.Contains(strings.ToUpper(imageFilename), "_WRONG") {
			isWrongFile = "TRUE"
		}

		// FRCNN PART
		cmdFRCNN := exec.Command("uv", "run", "python", frcnnScriptPath, imagePath, currentTempDir)
		frcnnStdOut, err := cmdFRCNN.Output()
		if err != nil {
			if exitErr, ok := err.(*exec.ExitError); ok {
				fmt.Printf("  [Warning] FRCNN script exited with error. Stderr: %s\n", string(exitErr.Stderr))
			} else {
				fmt.Printf("  [Warning] Error running FRCNN script: %v\n", err)
			}
			frcnnStdOut = []byte("0")
		}

		numFRCNNProposals, convErr := strconv.Atoi(strings.TrimSpace(string(frcnnStdOut)))
		if convErr != nil {
			numFRCNNProposals = 0
		}
		
		fmt.Printf("  FRCNN generated %d proposals for %s\n", numFRCNNProposals, imageFilename)

		finalCropIdxCounter := 0
		numToTakeFromFRCNN := 0
		
		if numFRCNNProposals >= totalCropsPerImage {
			numToTakeFromFRCNN = totalCropsPerImage
		} else if numFRCNNProposals > 0 {
			numToTakeFromFRCNN = numFRCNNProposals
		}

		frcnnMetaPath := filepath.Join(currentTempDir, "frcnn_meta.csv")
		if numToTakeFromFRCNN > 0 {
			fmt.Printf("  Taking top %d proposals from FRCNN for %s\n", numToTakeFromFRCNN, imageFilename)
			frcnnCrops, err := readTempMetaCSV(frcnnMetaPath, true)
			if err != nil {
				fmt.Printf("  [Warning] Could not read FRCNN meta CSV: %v\n", err)
			} else {
				for i := 0; i < len(frcnnCrops) && i < numToTakeFromFRCNN; i++ {
					crop := frcnnCrops[i]
					if err := processAndSaveCrop(imageFilename, finalCropIdxCounter, isWrongFile, crop, currentTempDir, "TRUE", "FALSE", csvWriter); err == nil {
						finalCropIdxCounter++
					}
				}
			}
		}
		csvWriter.Flush()

		// BING PART
		numBingNeeded := totalCropsPerImage - finalCropIdxCounter
		if numBingNeeded < 0 {
			numBingNeeded = 0
		}

		if numBingNeeded > 0 {
			absBingExecutablePath, _ := filepath.Abs(bingExecutablePath)
			absImagePath, _ := filepath.Abs(imagePath)
			absCurrentTempDir, _ := filepath.Abs(currentTempDir)

			cmdBING := exec.Command(absBingExecutablePath, absImagePath, strconv.Itoa(numBingNeeded), absCurrentTempDir)
			cmdBING.Dir = filepath.Dir(absBingExecutablePath)

			_, err := cmdBING.CombinedOutput()
			if err != nil {
				fmt.Printf("  [Warning] Error status from BING executable: %v\n", err)
			}

			bingMetaPath := filepath.Join(currentTempDir, "bing_meta.csv")
			if _, statErr := os.Stat(bingMetaPath); !os.IsNotExist(statErr) {
				bingCrops, errRead := readTempMetaCSV(bingMetaPath, false)
				if errRead == nil {
					processedBingCount := 0
					for i := 0; i < len(bingCrops) && processedBingCount < numBingNeeded; i++ {
						crop := bingCrops[i]
						if err := processAndSaveCrop(imageFilename, finalCropIdxCounter, isWrongFile, crop, currentTempDir, "FALSE", "TRUE", csvWriter); err == nil {
							finalCropIdxCounter++
							processedBingCount++
						}
					}
					fmt.Printf("  BING took %d proposals for %s\n", processedBingCount, imageFilename)
				}
			} else {
				fmt.Printf("  [Warning] BING meta file NOT FOUND.\n")
			}
		}
		csvWriter.Flush()

		// LOW-SALIENCY PART
		imgWidth, imgHeight, errDim := getImageDimensions(imagePath)
		if errDim == nil && imgWidth >= lowSaliencyCropTargetWidth && imgHeight >= lowSaliencyCropTargetHeight {
			generatedLowSaliencyCount := 0
			for i := 0; i < numLowSaliencyCrops; i++ {
				randX := rand.Intn(imgWidth - lowSaliencyCropTargetWidth + 1)
				randY := rand.Intn(imgHeight - lowSaliencyCropTargetHeight + 1)

				errCrop := generateAndSaveLowSaliencyCrop(
					imagePath,
					imageFilename,
					finalCropIdxCounter,
					isWrongFile,
					randX, randY, lowSaliencyCropTargetWidth, lowSaliencyCropTargetHeight,
					csvWriter,
				)
				if errCrop == nil {
					finalCropIdxCounter++
					generatedLowSaliencyCount++
				}
			}
			fmt.Printf("  Added %d low-saliency crops for %s\n", generatedLowSaliencyCount, imageFilename)
		} else if errDim != nil {
			fmt.Printf("  [Warning] Could not get dimensions: %v\n", errDim)
		} else {
			fmt.Printf("  [Warning] Image is too small for low-saliency target size.\n")
		}
		
		csvWriter.Flush()
		fmt.Printf("  Total crops for %s: %d\n", imageFilename, finalCropIdxCounter)
		
		os.RemoveAll(currentTempDir)
	}

	csvWriter.Flush()
	fmt.Println("\n--------------------------------------")
	fmt.Printf("Crops saved in: %s\n", finalCropsDir)
	fmt.Printf("CSV saved at: %s\n", finalCSVFile)
}

func readTempMetaCSV(filePath string, hasScore bool) ([]CropMeta, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open temp meta file %s: %w", filePath, err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	_, err = reader.Read()
	if err == io.EOF {
		return []CropMeta{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("failed to read header from %s: %w", filePath, err)
	}

	var crops []CropMeta
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			continue
		}

		expectedCols := 5
		if hasScore {
			expectedCols = 6
		}
		if len(record) < expectedCols {
			continue
		}

		var crop CropMeta
		crop.RelativeCropPath = record[0]
		crop.X, _ = strconv.Atoi(record[1])
		crop.Y, _ = strconv.Atoi(record[2])
		crop.Width, _ = strconv.Atoi(record[3])
		crop.Height, _ = strconv.Atoi(record[4])
		if hasScore {
			crop.Score, _ = strconv.ParseFloat(record[5], 64)
		}
		crops = append(crops, crop)
	}	

	return crops, nil
}

func processAndSaveCrop(
	originalFilename string,
	cropIdx int,
	isWrongFile string,
	meta CropMeta,
	tempBaseDir string,
	isFRCNNSource string,
	isBINGSource string,
	csvWriter *csv.Writer) error {

	finalCropFilename := fmt.Sprintf("%s_combo_crop%d.jpg", strings.TrimSuffix(originalFilename, filepath.Ext(originalFilename)), cropIdx)
	sourceCropPath := filepath.Join(tempBaseDir, meta.RelativeCropPath)
	destCropPath := filepath.Join(finalCropsDir, finalCropFilename)

	sourceFile, err := os.Open(sourceCropPath)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	destFile, err := os.Create(destCropPath)
	if err != nil {
		return err
	}
	defer destFile.Close()

	_, err = io.Copy(destFile, sourceFile)
	if err != nil {
		return err
	}

	x1, y1 := meta.X, meta.Y
	x_br, y_br := meta.X+meta.Width, meta.Y+meta.Height

	csvRecord := []string{
		originalFilename,
		strconv.Itoa(cropIdx),
		strconv.Itoa(x1), strconv.Itoa(y1),
		strconv.Itoa(x_br), strconv.Itoa(y1),
		strconv.Itoa(x1), strconv.Itoa(y_br),
		strconv.Itoa(x_br), strconv.Itoa(y_br),
		isWrongFile,
		isFRCNNSource,
		isBINGSource,
	}
	if err := csvWriter.Write(csvRecord); err != nil {
		return err
	}
	return nil
}