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
)

const totalCropsPerPainting = 20

// Config (adjust paths as needed, relative to where orchestrator.go is compiled and run from, i.e., main/)
var (
	paintingsDir = "../images/paintings" // Relative to main/
	bingProcessorDir         = "./bing_processor"
	frcnnProcessorDir        = "./frcnn_processor"
	bingExecutablePath       = filepath.Join(bingProcessorDir, "build", "BingCropperSingle") // Ensure this name matches CMake output
	frcnnScriptPath          = filepath.Join(frcnnProcessorDir, "src", "rp_rcnn_single.py")
	frcnnPythonVenvPath      = filepath.Join(frcnnProcessorDir, "venv_main_frcnn", "bin", "python") // Adjust if venv structure/name differs
	finalOutputDir           = "./output"
	finalCropsDir            = filepath.Join(finalOutputDir, "crops")
	finalCSVFile             = filepath.Join(finalOutputDir, "combined_data.csv")
	tempProcessingBaseDir    = "./temp_processing"
)

type CropMeta struct {
	RelativeCropPath string
	X                int
	Y                int
	Width            int
	Height           int
	Score            float64 // Only for FRCNN
}

func main() {
	log.Println("Orchestrator (Go): Starting...")

	// Setup output directories
	if err := os.MkdirAll(finalCropsDir, 0755); err != nil {
		log.Fatalf("Error creating final crops directory: %v", err)
	}
	if err := os.MkdirAll(tempProcessingBaseDir, 0755); err != nil {
		log.Fatalf("Error creating temp base directory: %v", err)
	}

	// Initialize final CSV
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
	csvWriter.Flush() // Write header immediately

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
		if !(strings.HasSuffix(strings.ToLower(paintingFilename), ".jpg") || strings.HasSuffix(strings.ToLower(paintingFilename), ".jpeg")) {
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
		defer os.RemoveAll(currentTempDir) // Cleanup temp dir for this image

		isWrongFile := "FALSE"
		if strings.Contains(strings.ToUpper(paintingFilename), "_WRONG") {
			isWrongFile = "TRUE"
		}

		// 1. Run FRCNN
		log.Printf("  Orchestrator: Running FRCNN for %s...\n", paintingFilename)
		cmdFRCNN := exec.Command(frcnnPythonVenvPath, frcnnScriptPath, paintingPath, currentTempDir)
		frcnnStdOut, err := cmdFRCNN.Output() // Captures standard output
		if err != nil {
			if exitErr, ok := err.(*exec.ExitError); ok {
				log.Printf("  FRCNN script for %s exited with error: %v. Stderr: %s\n", paintingFilename, err, string(exitErr.Stderr))
			} else {
				log.Printf("  Error running FRCNN script for %s: %v\n", paintingFilename, err)
			}
			frcnnStdOut = []byte("0") // Assume 0 proposals on error
		}

		numFRCNNProposals, convErr := strconv.Atoi(strings.TrimSpace(string(frcnnStdOut)))
		if convErr != nil {
			log.Printf("  Warning: FRCNN script for %s returned non-integer proposal count: '%s'. Assuming 0. Error: %v\n", paintingFilename, string(frcnnStdOut), convErr)
			numFRCNNProposals = 0
		}
		log.Printf("  Orchestrator: FRCNN generated %d proposals for %s.\n", numFRCNNProposals, paintingFilename)

		finalCropIdxCounter := 0

		// Consolidate FRCNN results
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
				// FRCNN script should ideally output sorted by score.
				// Here we just take the top N as they appear in its meta file.
				for i := 0; i < len(frcnnCrops) && i < numToTakeFromFRCNN; i++ {
					crop := frcnnCrops[i]
					if err := processAndSaveCrop(paintingFilename, finalCropIdxCounter, isWrongFile, crop, currentTempDir, "TRUE", "FALSE", csvWriter); err == nil {
						finalCropIdxCounter++
					}
				}
			}
		}
		csvWriter.Flush()


		// 2. Determine BING's task and Run BING if needed
		numBingNeeded := totalCropsPerPainting - finalCropIdxCounter
		if numBingNeeded < 0 { // Should not happen if FRCNN logic is correct
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

			bingCombinedOutput, err := cmdBING.CombinedOutput() // Captures both stdout and stderr
            
            // ALWAYS LOG BING'S OUTPUT for diagnosis
            log.Printf("  Raw BING Process Output for %s (length %d):\n---BEGIN BING STDERR/STDOUT---\n%s\n---END BING STDERR/STDOUT---\n", 
                        paintingFilename, len(bingCombinedOutput), string(bingCombinedOutput))

			if err != nil {
				// This log might now be redundant if the raw output above shows the error details
				log.Printf("  Error status from BING executable for %s: %v.\n", paintingFilename, err)
			}
            // else if len(bingCombinedOutput) == 0 {
			// 	log.Printf("  BING Process for %s produced no output to stdout/stderr but exited successfully.\n", paintingFilename)
			// }


			// Now, proceed to check the bing_meta.csv file as before
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
	_, err = reader.Read() // Skip header
	if err == io.EOF {
		return []CropMeta{}, nil // Empty file after header
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
			// Log malformed line and continue if possible
			log.Printf("Warning: Malformed line in %s: %v (line: %s)", filePath, err, strings.Join(record, ","))
			continue
		}

		if (hasScore && len(record) < 6) || (!hasScore && len(record) < 5) {
			log.Printf("Warning: Insufficient columns in line from %s: %s", filePath, strings.Join(record, ","))
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

	// Copy the crop file
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

	// Write to final CSV
	// tl_x,tl_y, tr_x,tr_y, bl_x,bl_y, br_x,br_y
	x1, y1 := meta.X, meta.Y
	x2, y2 := meta.X+meta.Width, meta.Y+meta.Height
	
	csvRecord := []string{
		originalFilename,
		strconv.Itoa(cropIdx),
		strconv.Itoa(x1), strconv.Itoa(y1), // top_left
		strconv.Itoa(x2), strconv.Itoa(y1), // top_right
		strconv.Itoa(x1), strconv.Itoa(y2), // bottom_left
		strconv.Itoa(x2), strconv.Itoa(y2), // bottom_right
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