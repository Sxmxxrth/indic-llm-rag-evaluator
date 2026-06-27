import os
import pandas as pd
from typing import List, Dict
from langchain.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM

# Mock LLM for demonstration purposes
class MockIndicLLM(LLM):
    def _call(self, prompt: str, stop: List[str] = None) -> str:
        # In production, this would call Sarvam API or OpenAI
        if "hallucination" in prompt.lower():
            return "SCORE: 0.95"
        return "SCORE: 0.88"
        
    @property
    def _llm_type(self) -> str:
        return "mock_indic_llm"

class IndicRAGEvaluator:
    """
    Evaluates RAG systems specifically on Hindi/Hinglish contexts.
    """
    def __init__(self, llm=None):
        self.llm = llm or MockIndicLLM()
        
        # Custom prompt for Hindi faithfulness evaluation
        self.faithfulness_prompt = PromptTemplate(
            template="""
            आप एक निष्पक्ष मूल्यांकनकर्ता (judge) हैं।
            नीचे दिए गए संदर्भ (Context) और उत्तर (Answer) को पढ़ें। 
            
            Context: {context}
            Answer: {answer}
            
            क्या उत्तर पूरी तरह से संदर्भ पर आधारित है? 
            Evaluate on a scale of 0.0 to 1.0 and output ONLY the score in format: SCORE: <value>
            """,
            input_variables=["context", "answer"]
        )
        
    def evaluate_faithfulness(self, context: str, answer: str) -> float:
        """Evaluates if the Hindi answer is faithful to the retrieved context."""
        formatted_prompt = self.faithfulness_prompt.format(context=context, answer=answer)
        response = self.llm.invoke(formatted_prompt)
        
        try:
            score = float(response.split("SCORE:")[1].strip())
            return score
        except Exception as e:
            print(f"Error parsing score: {e}")
            return 0.0

    def evaluate_batch(self, dataset: List[Dict]) -> pd.DataFrame:
        """Evaluates a batch of RAG outputs."""
        results = []
        for item in dataset:
            score = self.evaluate_faithfulness(item['context'], item['answer'])
            results.append({
                "question": item['question'],
                "faithfulness_score": score
            })
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Test Data
    test_data = [
        {
            "question": "Sarvam AI kya banata hai?",
            "context": "Sarvam AI is building full-stack AI for India, focusing on LLMs and voice agents for Indic languages.",
            "answer": "Sarvam AI भारत के लिए फुल-स्टैक AI बना रहा है, जिसमें इंडिक भाषाओं के लिए LLMs और वॉयस एजेंट शामिल हैं।"
        }
    ]
    
    evaluator = IndicRAGEvaluator()
    print("Evaluating Indic RAG Pipeline...")
    results_df = evaluator.evaluate_batch(test_data)
    print("\nEvaluation Results:")
    print(results_df.to_markdown(index=False))
