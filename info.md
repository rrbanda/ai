# What is AI (Artificial Intelligence)?

Artificial Intelligence (AI) is the simulation of human intelligence processes by computer systems. These processes include learning from experience, reasoning, problem-solving, understanding language, and adapting to new information. AI systems use a combination of algorithms, data, and computing power to carry out these tasks autonomously, often with an increasing level of complexity and capability.

AI can be broadly categorized into two types:

### Narrow AI vs General AI

| Type of AI                          | Description                                                                                                                                 | Key Features                                                                                               |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| Narrow AI (Weak AI)                 | Narrow AI is designed to perform specific tasks within a limited scope. Examples include Siri, Alexa, and Google Assistant.                 | Limited to specific tasks and rules, cannot handle complex issues or tasks outside its scope.              |
| General AI (Strong AI)              | General AI is a theoretical concept that aims to replicate human cognitive abilities capable of learning and performing intellectual tasks. | Able to learn, comprehend, and perform new tasks using prior knowledge without additional human training.   |

---

# What is Machine Learning (ML)?

Machine Learning (ML) is a branch of computer science that focuses on using data and algorithms to enable AI to imitate the way that humans learn, gradually improving its accuracy.

### How does Machine Learning work?

Machine Learning is a fundamental building block for more advanced AI applications, including both Predictive AI (making predictions based on historical data) and Generative AI (creating new content like text or images based on patterns in the data). Machine learning breaks down the learning system into three main parts:

1. **A Decision Process**: Machine learning algorithms make predictions or classifications based on input data.
2. **An Error Function**: Measures how far off the model’s predictions are from the target.
3. **A Model Optimization Process**: The model adjusts internal parameters to minimize the error, improving predictions.

Machine Learning identifies patterns in data and uses these patterns to make decisions or predictions.

### Types of Machine Learning:

- **Supervised Learning**: Training models on labeled data to make predictions or classifications.
- **Unsupervised Learning**: Identifying patterns in unlabeled data.
- **Reinforcement Learning**: Learning by interacting with an environment and optimizing decisions based on rewards or penalties.
  
### Deep Learning:

A type of machine learning that uses **neural networks** to model complex patterns in data, often achieving superior performance in tasks like language processing, vision, and more.

---

# What is a Neural Network?

A neural network is a machine learning model that mimics how the human brain works by using processes that resemble how biological neurons work together.

- Neural networks simulate the brain with layers of artificial neurons (input, hidden, and output layers).
- Each node has its own associated weight and threshold, determining whether it sends data to the next layer.

---

# Techniques Used in AI

- **Machine Learning (ML)**: AI technique where computers learn from data and improve over time.
- **Deep Learning**: Uses neural networks to make complex correlations in data.
- **Models**: Mathematical engines trained to perform tasks like prediction or content generation.

---

# Predictive AI vs Generative AI

| Type                      | Description                                                                                                                                                   | Examples                                                                                          |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| **Predictive AI**          | Predicts future outcomes based on historical data. It focuses on identifying patterns to forecast outcomes.                                                    | Customer behavior analysis, fraud detection, financial forecasting, predictive maintenance.        |
| **Generative AI**          | Generates new content based on the data it has been trained on. It creates new information, such as text, images, or music.                                     | Text generation, image creation, music generation.                                                |

---

# What is a Model in AI?

A **model** is a mathematical representation trained to perform a specific task, such as making predictions (predictive models) or creating new content (generative models).

---

# Relevance of AI and ML to Generative AI

Generative AI leverages machine learning models, particularly deep learning, to generate new content. It relies on neural networks to create outputs like text, images, or music, making it a specialized application of AI and ML.

---

# What is a Large Language Model (LLM)?

LLMs are neural network models trained on large-scale textual data to generate or understand human language. Examples include **GPT, BERT, and T5**.

---

# How is Generative AI and LLM Related?

LLMs are a subset of generative AI that focus specifically on text-based generation. They use **Transformer architecture** for processing and generating language effectively.

---

# What is GPT and Its Role in Generative AI?

**GPT (Generative Pre-trained Transformer)** is a type of LLM that uses Transformer architecture to generate human-like text. It’s widely used for tasks like conversation, content creation, and answering questions.

---

# Transformer-Based Models

Transformer-based models use **self-attention mechanisms** to weigh the importance of different words in a sequence. Popular models like **GPT, BERT, T5, and DALL-E** are based on Transformer architecture.

---

# Fine-Tuning vs Pre-Training

| Aspect          | Pre-Training                             | Fine-Tuning                                                                                     |
|-----------------|------------------------------------------|------------------------------------------------------------------------------------------------|
| **Purpose**     | General language understanding           | Adaptation to perform specific tasks.                                                          |
| **Learning**    | Unsupervised learning on vast datasets   | Supervised learning on a smaller, task-specific dataset.                                        |
| **Outcome**     | General-purpose model                    | Task-specific model, such as customer support or domain-specific queries.                        |
| **Time**        | Longer and resource-intensive            | Quicker, focusing on adjusting specific parameters.                                             |

---

# LoRA (Low-Rank Adaptation)

**LoRA** is a technique for fine-tuning large models efficiently by adjusting only a low-rank subset of model parameters. It reduces computational cost and time required for fine-tuning.

---

# Comparison of GPT, Llama, and Granite-20B Models

- **GPT-3**: 175 billion parameters.
- **Llama-3**: Up to 405 billion parameters with function-calling capabilities.
- **Granite-20B**: 20 billion parameters, optimized for structured responses and API interactions.

---

# MLOps Process

**MLOps** refers to the practice of integrating machine learning models into production environments. The process includes:

1. **Understanding the Use Case**
2. **Gathering and Preparing Data**
3. **Developing or Tuning the Model**
4. **Deploying the Model**
5. **Operating and Monitoring the Model**
6. **Retraining the Model**

---

# Prompt Engineering

**Prompt engineering** is the process of designing input prompts to guide a pre-trained language model to generate accurate and useful responses. It helps leverage model capabilities without needing to retrain it.

---

# Tokenization

Tokenization is the process of breaking down text into smaller components called tokens, which are processed by the model. Different tokenization methods include:

- **Word Tokenization**
- **Character Tokenization**
- **Subword Tokenization (e.g., BPE, WordPiece)**

---

# Parameters & Tokens

- **Parameters** are internal model weights adjusted during training.
- **Tokens** are the fundamental units of text processed by the model.

---

# Summary

AI and ML techniques, such as machine learning, neural networks, and fine-tuning, are at the core of both predictive and generative AI. Understanding models like **GPT, Llama, and Granite-20B**, along with methods like **LoRA** and **prompt engineering**, is essential for optimizing AI applications.
