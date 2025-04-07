import { users, type User, type InsertUser, datasets, type Dataset, type InsertDataset } from "@shared/schema";

// modify the interface with any CRUD methods
// you might need

export interface IStorage {
  // User methods
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  
  // Dataset methods
  getAllDatasets(): Promise<Dataset[]>;
  getDataset(id: number): Promise<Dataset | undefined>;
  createDataset(dataset: InsertDataset): Promise<Dataset>;
}

export class MemStorage implements IStorage {
  private users: Map<number, User>;
  private datasets: Map<number, Dataset>;
  private userIdCounter: number;
  private datasetIdCounter: number;

  constructor() {
    this.users = new Map();
    this.datasets = new Map();
    this.userIdCounter = 1;
    this.datasetIdCounter = 1;
  }

  // User methods
  async getUser(id: number): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = this.userIdCounter++;
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }
  
  // Dataset methods
  async getAllDatasets(): Promise<Dataset[]> {
    return Array.from(this.datasets.values());
  }
  
  async getDataset(id: number): Promise<Dataset | undefined> {
    return this.datasets.get(id);
  }
  
  async createDataset(insertDataset: InsertDataset): Promise<Dataset> {
    const id = this.datasetIdCounter++;
    // Ensure we convert undefined to null for filePath
    const filePath = insertDataset.filePath === undefined ? null : insertDataset.filePath;
    const dataset: Dataset = { 
      ...insertDataset, 
      id,
      filePath  // Explicitly setting filePath to handle undefined/null cases
    };
    this.datasets.set(id, dataset);
    return dataset;
  }
}

export const storage = new MemStorage();
