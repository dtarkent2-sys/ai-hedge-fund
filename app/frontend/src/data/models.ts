import { api } from '@/services/api';

export interface LanguageModel {
  display_name: string;
  model_name: string;
  provider: "Anthropic" | "DeepSeek" | "Google" | "Groq" | "OpenAI" | "Ollama" | string;
}

// Cache for models to avoid repeated API calls
let languageModels: LanguageModel[] | null = null;

/**
 * Get the list of models from the backend API
 * Uses caching to avoid repeated API calls
 */
export const getModels = async (): Promise<LanguageModel[]> => {
  if (languageModels) {
    return languageModels;
  }

  try {
    languageModels = await api.getLanguageModels();
    return languageModels;
  } catch (error) {
    console.error('Failed to fetch models:', error);
    throw error; // Let the calling component handle the error
  }
};

/**
 * Pick a sensible default model.
 *
 * Order:
 *   1. The first Ollama model (the backend only surfaces these when
 *      OLLAMA_API_KEY is set, so if any are present the user is on cloud Ollama)
 *   2. gpt-oss:20b / gpt-oss:120b if they showed up under any provider
 *   3. gpt-4.1 (legacy default)
 *   4. The first model in the list
 */
export const getDefaultModel = async (): Promise<LanguageModel | null> => {
  try {
    const models = await getModels();
    const ollama = models.find(m => m.provider === "Ollama");
    if (ollama) return ollama;
    const oss20 = models.find(m => m.model_name === "gpt-oss:20b");
    if (oss20) return oss20;
    const oss120 = models.find(m => m.model_name === "gpt-oss:120b");
    if (oss120) return oss120;
    return models.find(m => m.model_name === "gpt-4.1") || models[0] || null;
  } catch (error) {
    console.error('Failed to get default model:', error);
    return null;
  }
};
