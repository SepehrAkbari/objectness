package main

import (
	// "bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"image"
	"image/jpeg"
	_ "image/png"  // PNG dedoding support (now actually needed for our paintings)
	"math/rand"
	"time"
)

const totalCropsPerPainting = 20
const numLowSaliencyCrops = 5
const lowSaliencyCropTargetWidth = 224
const lowSaliencyCropTargetHeight = 224

var (
	paintingsDir = "../images/paintings"
	bingProcessorDir = "./bing_processor"
	frcnnProcessorDir = "./frcnn_processor"
	bingExecutablePath = filepath.Join(bingProcessorDir, "build", "BingCropperSingle")
	frcnnScriptPath = filepath.Join(frcnnProcessorDir, "src", "rp_rcnn_single.py")
	frcnnPythonVenvPath = filepath.Join(frcnnProcessorDir, "venv_main_frcnn", "bin", "python")
	finalOutputDir = "./output"
	finalCropsDir = filepath.Join(finalOutputDir, "crops")
	finalCSVFile = filepath.Join(finalOutputDir, "combined_data.csv")
	tempProcessingBaseDir = "./temp_processing"
)

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
	originalPaintingPath string,
	originalFilename string,
	cropIdx int,
	isWrongFile string,
	cropX, cropY, cropWidth, cropHeight int,
	csvWriter *csv.Writer) error {

	finalCropFilename := fmt.Sprintf("%s_lowsaliency_crop%d.jpg", strings.TrimSuffix(originalFilename, filepath.Ext(originalFilename)), cropIdx)
	destCropPath := filepath.Join(finalCropsDir, finalCropFilename)

	srcFile, err := os.Open(originalPaintingPath)
	if err != nil {
		log.Printf("  Error opening original painting %s for low-saliency crop: %v\n", originalPaintingPath, err)
		return err
	}
	defer srcFile.Close()

	img, _, err := image.Decode(srcFile)
	if err != nil {
		log.Printf("  Error decoding image %s for low-saliency crop: %v\n", originalPaintingPath, err)
		return err
	}

	cropRect := image.Rect(cropX, cropY, cropX+cropWidth, cropY+cropHeight)

	type subImager interface {
		SubImage(r image.Rectangle) image.Image
	}
	subImg, ok := img.(subImager)
	if !ok {
		log.Printf("  Error: image type does not support SubImage for %s\n", originalPaintingPath)
		return fmt.Errorf("image type does not support SubImage")
	}
	croppedImage := subImg.SubImage(cropRect)

	destFile, err := os.Create(destCropPath)
	if err != nil {
		log.Printf("  Error creating dest file %s for low-saliency crop: %v\n", destCropPath, err)
		return err
	}
	defer destFile.Close()

	err = jpeg.Encode(destFile, croppedImage, &jpeg.Options{Quality: 90})
	if err != nil {
		log.Printf("  Error encoding/saving low-saliency crop %s: %v\n", destCropPath, err)
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
		log.Printf("  Error writing record to final CSV for low-saliency crop %s: %v\n", originalFilename, err)
		return err
	}
	return nil
}


func main() {
	log.Println("Orchestrator (Go): Starting...")
	rand.Seed(time.Now().UnixNano())

	if err := os.MkdirAll(finalCropsDir, 0755); err != nil {
		log.Fatalf("Error creating final crops directory: %v", err)
	}
	if err := os.MkdirAll(tempProcessingBaseDir, 0755); err != nil {
		log.Fatalf("Error creating temp base directory: %v", err)
	}

	csvFile, err := os.Create(finalCSVFile)
	if err != nil {
		log.Fatalf("Error creating final CSV file: %v", err)
	}
	defer csvFile.Close()
	csvWriter := csv.NewWriter(csvFile)
	header := []string{"original_filename", "crop_idx", "top_left_x", "top_left_y", "top_right_x", "top_right_y", "bottom_left_x", "bottom_left_y", "bottom_right_x", "bottom_right_y", "WRONG_file", "FRCNN_source", "BING_source"}
	if err := csvWriter.Write(header); err != nil {
		log.Fatalf("Error writing CSV header: %v", err)
	}
	csvWriter.Flush()

	log.Printf("Orchestrator: Processing paintings from %s\n", paintingsDir)
	entries, err := os.ReadDir(paintingsDir)
	if err != nil {
		log.Fatalf("Error reading paintings directory: %v", err)
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		paintingFilename := entry.Name()
		if !(strings.HasSuffix(strings.ToLower(paintingFilename), ".jpg") || strings.HasSuffix(strings.ToLower(paintingFilename), ".jpeg") || strings.HasSuffix(strings.ToLower(paintingFilename), ".png")) {
			continue
		}

		paintingPath := filepath.Join(paintingsDir, paintingFilename)
		paintingBaseFilename := strings.TrimSuffix(paintingFilename, filepath.Ext(paintingFilename))
		log.Printf("Orchestrator: Processing painting: %s\n", paintingFilename)

		currentTempDir := filepath.Join(tempProcessingBaseDir, paintingBaseFilename+"_temp")
		if err := os.MkdirAll(filepath.Join(currentTempDir, "crops"), 0755); err != nil {
			log.Printf("  Error creating temp dir for %s: %v. Skipping.\n", paintingFilename, err)
			continue
		}
		defer os.RemoveAll(currentTempDir)

		isWrongFile := "FALSE"
		if strings.Contains(strings.ToUpper(paintingFilename), "_WRONG") {
			isWrongFile = "TRUE"
		}

		// FRCNN PART
		log.Printf("  Orchestrator: Running FRCNN for %s...\n", paintingFilename)
		cmdFRCNN := exec.Command(frcnnPythonVenvPath, frcnnScriptPath, paintingPath, currentTempDir)
		frcnnStdOut, err := cmdFRCNN.Output()
		if err != nil {
			if exitErr, ok := err.(*exec.ExitError); ok {
				log.Printf("  FRCNN script for %s exited with error: %v. Stderr: %s\n", paintingFilename, err, string(exitErr.Stderr))
			} else {
				log.Printf("  Error running FRCNN script for %s: %v\n", paintingFilename, err)
			}
			frcnnStdOut = []byte("0")
		}

		numFRCNNProposals, convErr := strconv.Atoi(strings.TrimSpace(string(frcnnStdOut)))
		if convErr != nil {
			log.Printf("  Warning: FRCNN script for %s returned non-integer proposal count: '%s'. Assuming 0. Error: %v\n", paintingFilename, string(frcnnStdOut), convErr)
			numFRCNNProposals = 0
		}
		log.Printf("  Orchestrator: FRCNN generated %d proposals for %s.\n", numFRCNNProposals, paintingFilename)

		finalCropIdxCounter := 0

		numToTakeFromFRCNN := 0
		if numFRCNNProposals >= totalCropsPerPainting {
			numToTakeFromFRCNN = totalCropsPerPainting
		} else if numFRCNNProposals > 0 {
			numToTakeFromFRCNN = numFRCNNProposals
		}

		frcnnMetaPath := filepath.Join(currentTempDir, "frcnn_meta.csv")
		if numToTakeFromFRCNN > 0 {
			log.Printf("  Orchestrator: Taking top %d proposals from FRCNN for %s.\n", numToTakeFromFRCNN, paintingFilename)
			frcnnCrops, err := readTempMetaCSV(frcnnMetaPath, true)
			if err != nil {
				log.Printf("  Warning: Could not read FRCNN meta CSV for %s: %v. Skipping FRCNN crops.\n", paintingFilename, err)
			} else {
				for i := 0; i < len(frcnnCrops) && i < numToTakeFromFRCNN; i++ {
					crop := frcnnCrops[i]
					if err := processAndSaveCrop(paintingFilename, finalCropIdxCounter, isWrongFile, crop, currentTempDir, "TRUE", "FALSE", csvWriter); err == nil {
						finalCropIdxCounter++
					}
				}
			}
		}
		csvWriter.Flush()

		// BING PART
		numBingNeeded := totalCropsPerPainting - finalCropIdxCounter
		if numBingNeeded < 0 {
			numBingNeeded = 0
		}

		if numBingNeeded > 0 {
			log.Printf("  Orchestrator: Running BING for %d proposals for %s...\n", numBingNeeded, paintingFilename)
			
			absBingExecutablePath, _ := filepath.Abs(bingExecutablePath)
			absPaintingPath, _ := filepath.Abs(paintingPath)
			absCurrentTempDir, _ := filepath.Abs(currentTempDir)

			cmdBING := exec.Command(absBingExecutablePath, absPaintingPath, strconv.Itoa(numBingNeeded), absCurrentTempDir)
			cmdBING.Dir = filepath.Dir(absBingExecutablePath)

			log.Printf("  Orchestrator: Executing BING: %s in CWD: %s\n", strings.Join(cmdBING.Args, " "), cmdBING.Dir)

			bingCombinedOutput, err := cmdBING.CombinedOutput()
			
			log.Printf("  Raw BING Process Output for %s (length %d):\n---BEGIN BING STDERR/STDOUT---\n%s\n---END BING STDERR/STDOUT---\n",
				paintingFilename, len(bingCombinedOutput), string(bingCombinedOutput))

			if err != nil {
				log.Printf("  Error status from BING executable for %s: %v.\n", paintingFilename, err)
			}

			bingMetaPath := filepath.Join(currentTempDir, "bing_meta.csv")
			if _, statErr := os.Stat(bingMetaPath); os.IsNotExist(statErr) {
				log.Printf("  Warning: BING meta file NOT FOUND for %s at %s (BING might have failed before creating it or crashed)\n", paintingFilename, bingMetaPath)
			} else {
				log.Printf("  Orchestrator: Attempting to take %d proposals from BING for %s from %s.\n", numBingNeeded, paintingFilename, bingMetaPath)
				bingCrops, errRead := readTempMetaCSV(bingMetaPath, false)
				if errRead != nil {
					log.Printf("  Warning: Could not read BING meta CSV for %s: %v. Skipping BING crops.\n", paintingFilename, errRead)
				} else {
					log.Printf("  Orchestrator: Read %d crop entries from BING meta file for %s.\n", len(bingCrops), paintingFilename)
					processedBingCount := 0
					for i := 0; i < len(bingCrops) && processedBingCount < numBingNeeded; i++ {
						crop := bingCrops[i]
						if err := processAndSaveCrop(paintingFilename, finalCropIdxCounter, isWrongFile, crop, currentTempDir, "FALSE", "TRUE", csvWriter); err == nil {
							finalCropIdxCounter++
							processedBingCount++
						}
					}
					log.Printf("  Orchestrator: Added %d crops from BING for %s.\n", processedBingCount, paintingFilename)
				}
			}
		}
		csvWriter.Flush()

		// LOW-SALIENCY PART
		log.Printf("  Orchestrator: Generating %d low-saliency crops for %s...\n", numLowSaliencyCrops, paintingFilename)
		imgWidth, imgHeight, errDim := getImageDimensions(paintingPath)
		if errDim != nil {
			log.Printf("  Warning: Could not get dimensions for %s: %v. Skipping low-saliency crops.\n", paintingFilename, errDim)
		} else {
			if imgWidth < lowSaliencyCropTargetWidth || imgHeight < lowSaliencyCropTargetHeight {
				log.Printf("  Warning: Image %s is smaller (%dx%d) than target low-saliency crop size (%dx%d). Skipping low-saliency crops.\n",
					paintingFilename, imgWidth, imgHeight, lowSaliencyCropTargetWidth, lowSaliencyCropTargetHeight)
			} else {
				generatedLowSaliencyCount := 0
				for i := 0; i < numLowSaliencyCrops; i++ {
					randX := rand.Intn(imgWidth - lowSaliencyCropTargetWidth + 1)
					randY := rand.Intn(imgHeight - lowSaliencyCropTargetHeight + 1)

					errCrop := generateAndSaveLowSaliencyCrop(
						paintingPath,
						paintingFilename,
						finalCropIdxCounter,
						isWrongFile,
						randX, randY, lowSaliencyCropTargetWidth, lowSaliencyCropTargetHeight,
						csvWriter,
					)
					if errCrop == nil {
						finalCropIdxCounter++
						generatedLowSaliencyCount++
					} else {
						log.Printf("  Failed to generate/save low-saliency crop #%d for %s: %v\n", i+1, paintingFilename, errCrop)
					}
				}
				log.Printf("  Orchestrator: Added %d low-saliency crops for %s.\n", generatedLowSaliencyCount, paintingFilename)
			}
		}
		csvWriter.Flush()


		log.Printf("  Orchestrator: Finished %s. Total crops generated for this image: %d\n", paintingFilename, finalCropIdxCounter)
		log.Println("--------------------------------------")
	}
	csvWriter.Flush()
	log.Println("Orchestration complete!")
	log.Printf("Final combined crops are in: %s\n", finalCropsDir)
	log.Printf("Final combined CSV is at: %s\n", finalCSVFile)
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
			log.Printf("Warning: Malformed line in %s: %v (line: %s)", filePath, err, strings.Join(record, ","))
			continue
		}

		expectedCols := 5
		if hasScore {
			expectedCols = 6
		}
		if len(record) < expectedCols {
			log.Printf("Warning: Insufficient columns in line from %s: expected %d, got %d (line: %s)", filePath, expectedCols, len(record), strings.Join(record, ","))
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
		log.Printf("  Error opening source crop %s: %v\n", sourceCropPath, err)
		return err
	}
	defer sourceFile.Close()

	destFile, err := os.Create(destCropPath)
	if err != nil {
		log.Printf("  Error creating dest crop %s: %v\n", destCropPath, err)
		return err
	}
	defer destFile.Close()

	_, err = io.Copy(destFile, sourceFile)
	if err != nil {
		log.Printf("  Error copying crop from %s to %s: %v\n", sourceCropPath, destCropPath, err)
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
		log.Printf("  Error writing record to final CSV for %s: %v\n", originalFilename, err)
		return err
	}
	return nil
}