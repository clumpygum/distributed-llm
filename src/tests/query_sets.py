query_sets = {
    "general_knowledge": [
        # Simple, shorter queries (Nano-friendly)
        "List the top five most significant scientific discoveries in the last 50 years and explain their impact.",
        "Explain very simply what happened during the Industrial Revolution.",
        "Briefly explain what artificial intelligence is.",
        "List 3 differences between Newtonian mechanics and general relativity.",  

        # Slightly more detailed but still manageable  
        "Explain the Big Bang theory in simple terms.",
        "Now go into more detail covering the formation of galaxies.",  

        # Long-form, structured outputs (More suitable for Orin)  
        "Generate a mock debate transcript for a scientist arguing against AI regulation.",
        "Expand on the AI regulation debate by adding a POV for AI regulation.",
        "Summarize everything we've discussed so far in one comprehensive report.",
        "Using all the context above, predict the future of AI in the next 50 years."
    ],
    "technical_coding": [
        "Explain the fundamentals of dynamic programming with an example.",
        "Now implement a Python function to solve the Knapsack problem.",
        "Optimize the previous function for better space complexity.",
        "Write a detailed guide on how blockchain technology works.",
        "Generate a Solidity smart contract for an NFT marketplace.",
        "Compare different LLM architectures (GPT, BERT, T5, etc.) with a focus on performance and applications.",
        "Generate a detailed technical walkthrough for building a distributed chatbot on Jetson devices.",
        "Given all previous context, draft a research paper proposal on optimizing LLM inference for edge devices.",
        "Using the full conversation history, create an executive summary of the research paper.",
        "Predict the future of edge AI computing and its impact on industries using all the context above."
    ],
    "personal_health": [
        # simpler
        "How often should adults get a general health checkup, and what does a typical checkup involve?",
        "What are five simple tips for improving my sleep quality?",
        "How can I tell if I'm adequately hydrated, and what’s a healthy daily water intake for an adult?",

        # more complex 
        "I'm looking to improve my diet to reduce inflammation. Could you provide a list of foods to avoid and foods I should include regularly, along with brief explanations?",
        "I work long hours seated at a desk, and I'm starting to experience lower back pain. Can you recommend specific exercises, stretches, and ergonomic adjustments to help alleviate and prevent this pain?",
        "I’ve been experiencing anxiety related to my workload and deadlines. Could you suggest practical stress management techniques and relaxation exercises that can be easily practiced at home or at work?",

        # complex
        "I’m experiencing frequent migraines that seem to worsen with stress, diet changes, and lack of sleep. Please provide a detailed analysis including potential triggers, dietary recommendations, lifestyle adjustments, and guidance on when to seek medical attention.",
        "Over the past few weeks, I've felt unusually fatigued, accompanied by mild dizziness and shortness of breath during mild exertion. Provide a structured symptom analysis, suggesting potential causes, recommended diagnostic steps, and the urgency of consulting a healthcare provider.",
        "I want to prepare for my first marathon in six months, but I have limited running experience. Generate a comprehensive and personalized six-month training plan including mileage progression, cross-training, injury prevention strategies, nutritional guidelines, and hydration recommendations.",
        "I’ve been feeling persistently sad, withdrawn from social activities, and unmotivated for over a month. Simulate a confidential mental health consultation."
    ]
}
