const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export interface SearchResult {
    id: string;
    text: string;
    relevance_score: number;
    confidence: "high" | "medium" | "low";
    rrf_score: number;
    bm25_score: number;
    dense_score: number;
    matched_concepts: string[];
    missing_concepts: string[];
}

export interface SearchResponse {
    query: string;
    results: SearchResult[];
    latency_ms: number;
    mode: string;
    confident: boolean;
}

export interface SynthesizeResponse {
    answer: string;
    grounded: boolean;
    warning: string | null;
}

export const api = {
    async search(query: string, mode: "hybrid" | "dense", top_k: number = 5): Promise<SearchResponse> {
        const res = await fetch(`${API_BASE}/search`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, mode, top_k })
        });
        if (!res.ok) throw new Error("Search failed");
        return res.json();
    },

    async synthesize(query: string, passages: string[]): Promise<SynthesizeResponse> {
        const res = await fetch(`${API_BASE}/synthesize`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, passages })
        });
        if (!res.ok) throw new Error("Synthesis failed");
        return res.json();
    },

    async submitFeedback(query: string, document_id: string, feedback: "relevant" | "not_relevant") {
        const res = await fetch(`${API_BASE}/feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, document_id, feedback })
        });
        if (!res.ok) throw new Error("Feedback submission failed");
        return res.json();
    },

    async addDocument(text: string) {
        const res = await fetch(`${API_BASE}/documents`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });
        if (!res.ok) throw new Error("Document upload failed");
        return res.json();
    }
};
