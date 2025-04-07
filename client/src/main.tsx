import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";
import { ComparisonProvider } from "./context/comparison-context";

createRoot(document.getElementById("root")!).render(
  <ComparisonProvider>
    <App />
  </ComparisonProvider>
);
