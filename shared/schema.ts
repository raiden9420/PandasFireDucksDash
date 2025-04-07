import { pgTable, text, serial, integer, boolean, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// User model for potential authentication in the future
export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;

// Dataset model
export const datasets = pgTable("datasets", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  rows: integer("rows").notNull(),
  columns: integer("columns").notNull(),
  sampleData: jsonb("sample_data").notNull(),
  createdAt: text("created_at").notNull(),
  filePath: text("file_path"),
});

export const insertDatasetSchema = createInsertSchema(datasets).omit({
  id: true,
});

export type InsertDataset = z.infer<typeof insertDatasetSchema>;
export type Dataset = typeof datasets.$inferSelect;

// Operations schema for determining which operations to run
export const Operations = z.object({
  load: z.boolean().default(true),
  groupby: z.boolean().default(true),
  merge: z.boolean().default(true),
  filter: z.boolean().default(true),
  rolling: z.boolean().default(true),
  // Advanced operations
  pivotTable: z.boolean().default(false),
  complexAggregation: z.boolean().default(false),
  windowFunctions: z.boolean().default(false),
  stringManipulation: z.boolean().default(false),
  nestedOperations: z.boolean().default(false),
  // Additional operations
  concat: z.boolean().default(false),
  sort: z.boolean().default(false),
  info: z.boolean().default(false),
  toCSV: z.boolean().default(false),
});

export type Operations = z.infer<typeof Operations>;

// Metrics schema for performance metrics
export const Metrics = z.object({
  executionTime: z.number(),
  memoryUsage: z.number(),
  operationTimes: z.record(z.string(), z.number()),
  version: z.string(),
  runs: z.array(z.object({
    executionTime: z.number(),
    memoryUsage: z.number(),
    operationTimes: z.record(z.string(), z.number()),
  })).optional(),
});

export type Metrics = z.infer<typeof Metrics>;

// Comparison results schema
export const Comparison = z.object({
  pandasMetrics: Metrics,
  fireducksMetrics: Metrics,
  resultsMatch: z.boolean(),
  datasetId: z.number().optional(),
});

export type Comparison = z.infer<typeof Comparison>;

// Generate Synthetic Data Request Schema
export const GenerateData = z.object({
  numRows: z.number().min(1000).max(1000000),
});

export type GenerateData = z.infer<typeof GenerateData>;

// Run Comparison Request Schema
export const RunComparison = z.object({
  datasetId: z.number().optional(),
  filePath: z.string().optional(),
  operations: Operations,
  settings: z.object({
    multiRunEnabled: z.boolean().default(false),
    runCount: z.number().min(1).max(10).default(3),
  }),
});

export type RunComparison = z.infer<typeof RunComparison>;
