import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import multer from "multer";
import { randomUUID } from "crypto";
import path from "path";
import fs from "fs";
import { GenerateData, RunComparison } from "@shared/schema";
import { 
  runAnalysis, 
  generateSyntheticData, 
  analyzeCSV 
} from "./controllers/analyzer";

// Configure multer for file uploads
const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => {
      const uploadDir = path.join(process.cwd(), "uploads");
      
      // Create uploads directory if it doesn't exist
      if (!fs.existsSync(uploadDir)) {
        fs.mkdirSync(uploadDir, { recursive: true });
      }
      
      cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
      // Generate unique filename
      const uniqueName = `${randomUUID()}-${file.originalname}`;
      cb(null, uniqueName);
    }
  }),
  limits: {
    fileSize: 50 * 1024 * 1024 // 50MB max file size
  },
  fileFilter: (req, file, cb) => {
    // Only accept CSV files
    if (file.mimetype === "text/csv" || file.originalname.endsWith(".csv")) {
      cb(null, true);
    } else {
      cb(new Error("Only CSV files are allowed"));
    }
  }
});

export async function registerRoutes(app: Express): Promise<Server> {
  // API Routes
  
  // Get all datasets
  app.get("/api/datasets", async (req, res) => {
    try {
      const datasets = await storage.getAllDatasets();
      res.json(datasets);
    } catch (error) {
      console.error("Error fetching datasets:", error);
      res.status(500).json({ message: error instanceof Error ? error.message : "Internal server error" });
    }
  });
  
  // Get dataset by ID
  app.get("/api/datasets/:id", async (req, res) => {
    try {
      const dataset = await storage.getDataset(parseInt(req.params.id));
      if (!dataset) {
        return res.status(404).json({ message: "Dataset not found" });
      }
      res.json(dataset);
    } catch (error) {
      console.error("Error fetching dataset:", error);
      res.status(500).json({ message: error instanceof Error ? error.message : "Internal server error" });
    }
  });
  
  // Upload CSV file
  app.post("/api/datasets/upload", upload.single("file"), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ message: "No file uploaded" });
      }
      
      const filePath = req.file.path;
      const fileName = path.basename(req.file.originalname, path.extname(req.file.originalname));
      
      // Analyze CSV file
      const analysis = await analyzeCSV(filePath);
      
      // Create dataset record
      const dataset = await storage.createDataset({
        name: fileName,
        rows: analysis.rows,
        columns: analysis.columns,
        sampleData: analysis.sample_data,
        createdAt: new Date().toISOString(),
        filePath: filePath,
      });
      
      res.json(dataset);
    } catch (error) {
      console.error("Error uploading file:", error);
      res.status(500).json({ message: error instanceof Error ? error.message : "Internal server error" });
    }
  });
  
  // Generate synthetic data
  app.post("/api/datasets/generate", async (req, res) => {
    try {
      const { numRows } = req.body as GenerateData;
      
      if (!numRows || numRows < 1000 || numRows > 1000000) {
        return res.status(400).json({ message: "Invalid number of rows (must be between 1,000 and 1,000,000)" });
      }
      
      // Generate synthetic data
      const data = await generateSyntheticData(numRows);
      
      // Create dataset record with the filePath from the generated data
      const dataset = await storage.createDataset({
        name: `Synthetic Data (${numRows.toLocaleString()} rows)`,
        rows: data.rows,
        columns: data.columns,
        sampleData: data.sample_data,
        createdAt: new Date().toISOString(),
        filePath: data.filePath, // This will be a real file path now
      });
      
      res.json(dataset);
    } catch (error) {
      console.error("Error generating synthetic data:", error);
      res.status(500).json({ message: error instanceof Error ? error.message : "Internal server error" });
    }
  });
  
  // Run comparison
  app.post("/api/comparison/run", async (req, res) => {
    try {
      const { datasetId, operations, settings } = req.body as RunComparison;
      
      if (!datasetId) {
        return res.status(400).json({ message: "Dataset ID is required" });
      }
      
      // Get dataset
      const dataset = await storage.getDataset(datasetId);
      if (!dataset) {
        return res.status(404).json({ message: "Dataset not found" });
      }
      
      // Run analysis
      const results = await runAnalysis(datasetId, operations, settings);
      
      res.json(results);
    } catch (error) {
      console.error("Error running comparison:", error);
      res.status(500).json({ message: error instanceof Error ? error.message : "Internal server error" });
    }
  });

  const httpServer = createServer(app);

  return httpServer;
}
