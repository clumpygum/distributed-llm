query_sets = {
    "general_knowledge": [
        # Simple, conversational queries (Nano-friendly)
        {"query": "Thank you!", "expected_device": "nano"},
        {"query": "Thanks for the help.", "expected_device": "nano"},
        
        # Simple, shorter queries (Nano-friendly)
        {"query": "Explain very simply what happened during the Industrial Revolution.", "expected_device": "nano"},
        {"query": "Briefly explain what artificial intelligence is.", "expected_device": "nano"},
        {"query": "List 3 differences between Newtonian mechanics and general relativity.", "expected_device": "nano"},  
        {"query": "Explain the Big Bang theory in simple terms.", "expected_device": "nano"},

        # Slightly more detailed / Long-form, structured outputs (More suitable for Orin)  
        {"query": "List the top five most significant scientific discoveries in the last 50 years and explain their impact.", "expected_device": "orin"},
        {"query": "Now go into more detail covering the formation of galaxies.", "expected_device": "orin"},  
        {"query": "Generate a mock debate transcript for a scientist arguing against AI regulation.", "expected_device": "orin"},
        {"query": "Expand on the AI regulation debate by adding a POV for AI regulation.", "expected_device": "orin"},
        {"query": "Summarize everything we've discussed so far in one comprehensive report.", "expected_device": "orin"},
        {"query": "Using all the context above, predict the future of AI in the next 50 years.", "expected_device": "orin"}
    ],
    "technical_coding": [
        # All of these require high reasoning, coding knowledge, or long context generation (Orin)
        {"query": "Explain the fundamentals of dynamic programming with an example.", "expected_device": "orin"},
        {"query": "Now implement a Python function to solve the Knapsack problem.", "expected_device": "orin"},
        {"query": "Optimize the previous function for better space complexity.", "expected_device": "orin"},
        {"query": "Write a detailed guide on how blockchain technology works.", "expected_device": "orin"},
        {"query": "Generate a Solidity smart contract for an NFT marketplace.", "expected_device": "orin"},
        {"query": "Compare different LLM architectures (GPT, BERT, T5, etc.) with a focus on performance and applications.", "expected_device": "orin"},
        {"query": "Generate a detailed technical walkthrough for building a distributed chatbot on Jetson devices.", "expected_device": "orin"},
        {"query": "Given all previous context, draft a research paper proposal on optimizing LLM inference for edge devices.", "expected_device": "orin"},
        {"query": "Using the full conversation history, create an executive summary of the research paper.", "expected_device": "orin"},
        {"query": "Predict the future of edge AI computing and its impact on industries using all the context above.", "expected_device": "orin"}
    ],
    "personal_health": [
        # Simpler factual questions (Nano)
        {"query": "How often should adults get a general health checkup, and what does a typical checkup involve?", "expected_device": "nano"},
        {"query": "What are five simple tips for improving my sleep quality?", "expected_device": "nano"},
        {"query": "How can I tell if I'm adequately hydrated, and what’s a healthy daily water intake for an adult?", "expected_device": "nano"},

        # More complex / Detailed planning (Orin)
        {"query": "I'm looking to improve my diet to reduce inflammation. Could you provide a list of foods to avoid and foods I should include regularly, along with brief explanations?", "expected_device": "orin"},
        {"query": "I work long hours seated at a desk, and I'm starting to experience lower back pain. Can you recommend specific exercises, stretches, and ergonomic adjustments to help alleviate and prevent this pain?", "expected_device": "orin"},
        {"query": "I’ve been experiencing anxiety related to my workload and deadlines. Could you suggest practical stress management techniques and relaxation exercises that can be easily practiced at home or at work?", "expected_device": "orin"},

        # Complex medical analysis and multi-step plans (Orin)
        {"query": "I’m experiencing frequent migraines that seem to worsen with stress, diet changes, and lack of sleep. Please provide a detailed analysis including potential triggers, dietary recommendations, lifestyle adjustments, and guidance on when to seek medical attention.", "expected_device": "orin"},
        {"query": "Over the past few weeks, I've felt unusually fatigued, accompanied by mild dizziness and shortness of breath during mild exertion. Provide a structured symptom analysis, suggesting potential causes, recommended diagnostic steps, and the urgency of consulting a healthcare provider.", "expected_device": "orin"},
        {"query": "I want to prepare for my first marathon in six months, but I have limited running experience. Generate a comprehensive and personalized six-month training plan including mileage progression, cross-training, injury prevention strategies, nutritional guidelines, and hydration recommendations.", "expected_device": "orin"},
        {"query": "I’ve been feeling persistently sad, withdrawn from social activities, and unmotivated for over a month. Simulate a confidential mental health consultation.", "expected_device": "orin"}
    ]
}