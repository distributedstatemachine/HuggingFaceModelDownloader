package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	// "path/filepath"
	"sort"
	"strings"
)

// Copy the required types from hfdownloader
type hfmodel struct {
	Type          string `json:"type"`
	Oid           string `json:"oid"`
	Size          int    `json:"size"`
	Path          string `json:"path"`
	LocalSize     int64
	NeedsDownload bool
	IsDirectory   bool
	IsLFS         bool
	Lfs           *hflfs `json:"lfs,omitempty"`
}

type hflfs struct {
	Oid_SHA265  string `json:"oid"`
	Size        int64  `json:"size"`
	PointerSize int    `json:"pointerSize"`
}

type FileStats struct {
	Name      string
	FileCount int
	TotalSize int64
}

// Copy formatSize helper function
func formatSize(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

////////////////////////////////////////////////////////////////////////////////
// BFS approach to get all files below a given path via Hugging Face API
////////////////////////////////////////////////////////////////////////////////

func fetchAllFiles(modelName, branch, basePath string) ([]hfmodel, error) {
	var result []hfmodel
	queue := []string{basePath}

	for len(queue) > 0 {
		dir := queue[0]
		queue = queue[1:]

		apiURL := fmt.Sprintf("https://huggingface.co/api/datasets/%s/tree/%s/%s",
			modelName, branch, dir)

		files, err := fetchFileList(apiURL)
		if err != nil {
			return nil, fmt.Errorf("failed to list %s: %v", dir, err)
		}
		for _, f := range files {
			if f.Type == "directory" {
				queue = append(queue, f.Path)
			}
			result = append(result, f)
		}
	}
	return result, nil
}

////////////////////////////////////////////////////////////////////////////////
// Main analysis function
////////////////////////////////////////////////////////////////////////////////

func analyzeDataset(modelName, branch string) error {
	fmt.Printf("Fetching everything ...\n")

	all, err := fetchAllFiles(modelName, branch, "/")
	if err != nil {
		return fmt.Errorf("failed BFS fetch: %v", err)
	}

	if len(all) == 0 {
		return fmt.Errorf("no items found")
	}

	// 1) Summarize total data folder
	var dataFilesCount int
	var dataTotalSize int64

	// 2) Keep track of every folder (not just CC-MAIN)
	// Map: folderPath -> FileStats
	folderStats := make(map[string]*FileStats)

	for _, item := range all {
		// If directory, skip size
		if item.Type == "directory" {
			// Ensure this folder is labeled in folderStats
			if _, ok := folderStats[item.Path]; !ok {
				folderStats[item.Path] = &FileStats{Name: item.Path}
			}
			continue
		}

		// item is a file
		dataFilesCount++

		var size int64 = int64(item.Size)
		if item.Lfs != nil && item.Lfs.Size > 0 {
			size = item.Lfs.Size
		}
		dataTotalSize += size

		// Identify its folder
		parts := strings.Split(item.Path, "/")
		if len(parts) > 1 {
			// For example: data/CC-MAIN-2013-20/something
			folderPath := strings.Join(parts[:len(parts)-1], "/")
			if _, ok := folderStats[folderPath]; !ok {
				folderStats[folderPath] = &FileStats{Name: folderPath}
			}
			fs := folderStats[folderPath]
			fs.FileCount++
			fs.TotalSize += size
		}
	}

	// Sort folderStats by their path
	sortedPaths := make([]string, 0, len(folderStats))
	for k := range folderStats {
		sortedPaths = append(sortedPaths, k)
	}
	sort.Strings(sortedPaths)

	////////////////////////////////////////////////////
	// Output
	fmt.Println("\n=== Dataset Analysis ===")
	fmt.Printf("Dataset: %s\n", modelName)
	fmt.Printf("Branch: %s\n", branch)

	fmt.Println("\nfolder overview:")
	fmt.Println("-------------------------------------------------------")
	fmt.Printf("Total files (including subfolders): %d\n", dataFilesCount)
	fmt.Printf("Total size: %s\n", formatSize(dataTotalSize))

	fmt.Println("\nAll Folders:")
	fmt.Println("-------------------------------------------------------")
	for _, p := range sortedPaths {
		stats := folderStats[p]
		fmt.Printf("Folder: %s\n", p)
		fmt.Printf("  Files: %d\n", stats.FileCount)
		fmt.Printf("  Size: %s\n", formatSize(stats.TotalSize))
		fmt.Println()
	}
	return nil
}

////////////////////////////////////////////////////////////////////////////////
// Helper: fetchFileList, formatSize
////////////////////////////////////////////////////////////////////////////////

func fetchFileList(url string) ([]hfmodel, error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed create request: %v", err)
	}

	req.Header.Add("User-Agent", "Mozilla/5.0")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch file list: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("status %d: %s", resp.StatusCode, string(body))
	}

	var files []hfmodel
	if err := json.NewDecoder(resp.Body).Decode(&files); err != nil {
		return nil, fmt.Errorf("failed JSON decode: %v", err)
	}

	return files, nil
}

func main() {
	modelName := "mlfoundations/dclm-baseline-1.0-parquet"
	branch := "main"
	if err := analyzeDataset(modelName, branch); err != nil {
		fmt.Printf("Error: %v\n", err)
	}
}
