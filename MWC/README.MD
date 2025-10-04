asdfdfStartup Similarity Analysis: MWC 2025 – Josep Cubedo Macian

1. Introduction
Objective
This project analyzes semantic similarity between startups attending MWC 2025 based on their textual descriptions. By computing similarity scores using advanced NLP techniques, we identify:
• Clusters of related startups
• Potential partnerships
• Emerging industry trends
Context
MWC (Mobile World Congress) is one of the largest global events for mobile and tech startups. Understanding how startups relate to each other provides valuable insights for:
• Networking opportunities
• Identifying potential competitors or collaborators
• Helping investors discover innovative startups in their areas of interest
 
2. Data Collection
Source of Data
The dataset consists of startup descriptions collected from MWC 2025 participants through interviews with official company representatives.
Dataset Structure
The dataset consists of a csv file with 37 entries, one per startup. Each entry contains:
Startup Name
Industry Category
Description (main text used for similarity analysis)
Main technology

Preprocessing Steps
• Text tokenization using BERT’s tokenizer
• Vectorization using BERT embeddings to capture semantic meaning
• Normalization of similarity scores
 
3. Methodology
Text Processing & Similarity Computation
The similarity between startups is computed using:
• BERT Embeddings: Converting startup descriptions into dense vector representations
• Cosine Similarity: Computing similarity between vector representations
• Normalization: Scaling similarity scores to a [0,1] range

Visualization Strategy
Three complementary visualization approaches were used:
1. Network Graph: Shows relationships between startups as a force-directed graph
2. Heatmap: Provides a comprehensive view of all pairwise similarities
 
4. Results & Findings
Network Visualization 
n=37, threshold=0.77, 
The network graph clusters startups by industry category and connects those with high similarity:
• Nodes represent startups, colored by industry category
• Edges connect startups with similarity above a threshold (0.77)
• Edge thickness indicates degree of similarity
• Layout algorithm positions similar startups closer together
Edges are stablished based on a defined threshold. In other words, only two companies with similarity above the threshold defined will connect. 
Heatmap Insights
 
The heatmap provides a detailed view of all pairwise similarities:
• Bright spots indicate highly similar startups
• Visible clusters suggest industry groupings
• Diagonal represents self-similarity (always 1.0)
 
5. Implementation Details
Dependencies
• numpy and torch for numerical computation
• transformers for BERT model and tokenizer
• sklearn for similarity metrics
• matplotlib and networkx for visualization

Key Features

Semantic Analysis: Uses BERT embeddings to capture meaning beyond keywords
Clustering Algorithm: Implements a custom layout algorithm that positions similar startups closer together
Anti-overlap Logic: Ensures nodes don’t overlap for better readability
Multi-format Visualization: Provides three complementary views of the similarity data
Limitations
Performance depends on the quality of startup descriptions
BERT embeddings capture semantic meaning but may miss specialized technical terminology
Current visualizations have scalability limitations for very large datasets

Future Enhancements
 Incorporate domain-specific language models for better technical understanding
Add interactive visualization capabilities for dynamic exploration
 Integrate additional data sources (funding, patents, market traction)
Implement temporal analysis to track startup evolution over time

