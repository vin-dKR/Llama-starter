import { HfInference } from '@huggingface/inference';
import dotenv from 'dotenv'

dotenv.config()

const apiKey: string = process.env.LLMA_API_KEY as string; // Replace with your actual Hugging Face API key
const hf = new HfInference(apiKey);

const systemPrompt = "You are a travel agent so tell your client descriptive and helpful about the places.";
const userPrompt = "Tell me about India in 10 lines.";

// console.log("API Key:", process.env.LLMA_API_KEY);

const main = async (): Promise<void> => {
	try {
		const response = await hf.textGeneration({
			model: 'meta-llama/Llama-3.2-11B-Vision-Instruct', // free-version
			inputs: `${systemPrompt}\n${userPrompt}`,
			parameters: {
				max_length: 1024,
				temperature: 0.7,
				top_k: 50,
				top_p: 0.95
			},
		});

		console.log("User:", userPrompt);
		console.log("AI:", response.generated_text);
	} catch (error) {
		console.error("Error during inference:", error);
	}
}

main();
