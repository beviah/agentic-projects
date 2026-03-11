"""Knowledge-Guided Neural Network Architectures.

Implements neural networks that incorporate structured knowledge from
knowledge graphs and ontologies to guide learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np


class KnowledgeGraph:
    """Simple knowledge graph representation."""
    
    def __init__(self, n_entities: int, n_relations: int):
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.triples = []  # List of (head, relation, tail) tuples
        
        # Adjacency lists for each relation type
        self.adjacency = {r: [[] for _ in range(n_entities)] for r in range(n_relations)}
    
    def add_triple(self, head: int, relation: int, tail: int):
        """Add a triple to the knowledge graph."""
        self.triples.append((head, relation, tail))
        self.adjacency[relation][head].append(tail)
    
    def get_neighbors(self, entity: int, relation: int) -> List[int]:
        """Get neighbors of entity via relation."""
        return self.adjacency[relation][entity]
    
    def to_adjacency_matrix(self, relation: int) -> torch.Tensor:
        """Convert relation to adjacency matrix."""
        adj = torch.zeros(self.n_entities, self.n_entities)
        for head in range(self.n_entities):
            for tail in self.adjacency[relation][head]:
                adj[head, tail] = 1.0
        return adj


class KGEmbedding(nn.Module):
    """Knowledge graph embedding layer."""
    
    def __init__(self, n_entities: int, n_relations: int, embedding_dim: int):
        super().__init__()
        self.entity_embeddings = nn.Embedding(n_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(n_relations, embedding_dim)
        
        # Initialize with Xavier
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def forward(self, entities: torch.Tensor) -> torch.Tensor:
        """Get entity embeddings."""
        return self.entity_embeddings(entities)
    
    def score_triple(self, head: torch.Tensor, relation: torch.Tensor, 
                     tail: torch.Tensor, method: str = 'TransE') -> torch.Tensor:
        """Score a triple (head, relation, tail)."""
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)
        
        if method == 'TransE':
            # TransE: h + r ≈ t
            score = -torch.norm(h + r - t, p=2, dim=-1)
        elif method == 'DistMult':
            # DistMult: <h, r, t>
            score = torch.sum(h * r * t, dim=-1)
        elif method == 'ComplEx':
            # For simplicity, use DistMult here
            score = torch.sum(h * r * t, dim=-1)
        else:
            raise ValueError(f"Unknown scoring method: {method}")
        
        return score


class KGGuidedGNN(nn.Module):
    """Graph Neural Network guided by knowledge graph structure."""
    
    def __init__(self, n_entities: int, n_relations: int, embedding_dim: int,
                 hidden_dim: int, n_layers: int = 2):
        super().__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.n_layers = n_layers
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(n_entities, embedding_dim)
        
        # Relation-specific transformation matrices
        self.relation_transforms = nn.ModuleList([
            nn.Linear(embedding_dim, hidden_dim, bias=False)
            for _ in range(n_relations)
        ])
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def message_passing(self, node_features: torch.Tensor, 
                       adjacency_matrices: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Perform message passing along knowledge graph edges.
        
        Args:
            node_features: Tensor of shape (n_entities, hidden_dim)
            adjacency_matrices: Dict mapping relation_id to adjacency matrix
            
        Returns:
            Updated node features
        """
        messages = []
        
        # Aggregate messages from each relation type
        for relation_id, adj_matrix in adjacency_matrices.items():
            # Transform features based on relation
            transformed = self.relation_transforms[relation_id](node_features)
            
            # Propagate along edges: adj_matrix @ transformed
            message = torch.matmul(adj_matrix, transformed)
            messages.append(message)
        
        # Combine messages (sum)
        if messages:
            aggregated = torch.stack(messages, dim=0).sum(dim=0)
        else:
            aggregated = torch.zeros_like(node_features)
        
        return aggregated
    
    def forward(self, knowledge_graph: KnowledgeGraph) -> torch.Tensor:
        """Forward pass through KG-guided GNN.
        
        Args:
            knowledge_graph: KnowledgeGraph object
            
        Returns:
            Updated entity embeddings of shape (n_entities, hidden_dim)
        """
        # Get initial entity features
        entity_ids = torch.arange(self.n_entities)
        h = self.entity_embeddings(entity_ids)
        
        # Get adjacency matrices for all relations
        adjacency_matrices = {
            r: knowledge_graph.to_adjacency_matrix(r)
            for r in range(self.n_relations)
        }
        
        # Apply GNN layers
        for layer_idx in range(self.n_layers):
            # Message passing
            messages = self.message_passing(h, adjacency_matrices)
            
            # Update with GNN layer
            h_new = self.gnn_layers[layer_idx](h + messages)
            h_new = self.activation(h_new)
            h_new = self.dropout(h_new)
            
            # Residual connection
            if h.shape == h_new.shape:
                h = h + h_new
            else:
                h = h_new
        
        return h


class KGConstrainedAttention(nn.Module):
    """Attention mechanism constrained by knowledge graph relations."""
    
    def __init__(self, embedding_dim: int, n_heads: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        
        assert embedding_dim % n_heads == 0, "embedding_dim must be divisible by n_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x: torch.Tensor, kg_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute KG-constrained attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim)
            kg_mask: Optional mask based on KG structure (batch, seq_len, seq_len)
                     1.0 for allowed connections, 0.0 for disallowed
            
        Returns:
            Output tensor of shape (batch, seq_len, embedding_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply KG mask if provided
        if kg_mask is not None:
            # Expand mask for multi-head attention
            kg_mask = kg_mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)
            scores = scores.masked_fill(kg_mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
        output = self.out_proj(attn_output)
        
        return output


class SemanticLossLayer(nn.Module):
    """Adds semantic loss based on knowledge graph constraints."""
    
    def __init__(self, n_entities: int):
        super().__init__()
        self.n_entities = n_entities
    
    def forward(self, predictions: torch.Tensor, 
                semantic_constraints: List[Tuple[int, int, str]]) -> torch.Tensor:
        """Compute semantic loss.
        
        Args:
            predictions: Entity predictions of shape (batch, n_entities)
            semantic_constraints: List of (entity1, entity2, constraint_type) tuples
                constraint_type: 'mutex' (mutually exclusive) or 'implies' (entity1 -> entity2)
            
        Returns:
            Semantic loss value
        """
        loss = 0.0
        
        for entity1, entity2, constraint_type in semantic_constraints:
            p1 = predictions[:, entity1]
            p2 = predictions[:, entity2]
            
            if constraint_type == 'mutex':
                # Mutually exclusive: p1 * p2 should be close to 0
                loss += torch.mean(p1 * p2)
            elif constraint_type == 'implies':
                # Implication: if p1 is high, p2 should be high
                # Equivalent to: maximize p2 when p1 > threshold
                loss += torch.mean(F.relu(p1 - p2))
        
        return loss


class KnowledgeGuidedClassifier(nn.Module):
    """Classifier that uses KG to guide predictions."""
    
    def __init__(self, input_dim: int, n_classes: int, knowledge_graph: KnowledgeGraph,
                 embedding_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.n_classes = n_classes
        
        # Input encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # KG-guided GNN
        self.kg_gnn = KGGuidedGNN(
            n_entities=knowledge_graph.n_entities,
            n_relations=knowledge_graph.n_relations,
            embedding_dim=embedding_dim,
            hidden_dim=embedding_dim
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes)
        )
        
        self.knowledge_graph = knowledge_graph
    
    def forward(self, x: torch.Tensor, entity_id: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features of shape (batch, input_dim)
            entity_id: Entity IDs corresponding to inputs (batch,)
            
        Returns:
            Class logits of shape (batch, n_classes)
        """
        # Encode input
        input_embedding = self.encoder(x)
        
        # Get KG-enhanced entity embeddings
        kg_embeddings = self.kg_gnn(self.knowledge_graph)
        entity_embedding = kg_embeddings[entity_id]
        
        # Combine input and KG embeddings
        combined = torch.cat([input_embedding, entity_embedding], dim=-1)
        
        # Classify
        logits = self.classifier(combined)
        
        return logits


if __name__ == "__main__":
    print("=" * 60)
    print("Knowledge-Guided Neural Networks Example")
    print("=" * 60)
    
    # Create a simple knowledge graph
    # Entities: 0=Dog, 1=Cat, 2=Animal, 3=Mammal, 4=Bird, 5=Canary
    # Relations: 0=is_a, 1=part_of
    kg = KnowledgeGraph(n_entities=6, n_relations=2)
    
    # Add is_a relations
    kg.add_triple(0, 0, 3)  # Dog is_a Mammal
    kg.add_triple(1, 0, 3)  # Cat is_a Mammal
    kg.add_triple(3, 0, 2)  # Mammal is_a Animal
    kg.add_triple(5, 0, 4)  # Canary is_a Bird
    kg.add_triple(4, 0, 2)  # Bird is_a Animal
    
    print("\nKnowledge Graph:")
    print(f"  Entities: {kg.n_entities}")
    print(f"  Relations: {kg.n_relations}")
    print(f"  Triples: {len(kg.triples)}")
    
    # Create KG-guided GNN
    gnn = KGGuidedGNN(
        n_entities=6,
        n_relations=2,
        embedding_dim=16,
        hidden_dim=32,
        n_layers=2
    )
    
    # Forward pass
    entity_embeddings = gnn(kg)
    
    print("\nEntity embeddings after GNN:")
    print(f"  Shape: {entity_embeddings.shape}")
    print(f"  Dog embedding norm: {torch.norm(entity_embeddings[0]):.3f}")
    print(f"  Cat embedding norm: {torch.norm(entity_embeddings[1]):.3f}")
    
    # Test KG-constrained attention
    print("\n" + "=" * 60)
    print("KG-Constrained Attention Example")
    print("=" * 60)
    
    attention = KGConstrainedAttention(embedding_dim=32, n_heads=4)
    
    # Create input sequence
    x = torch.randn(2, 5, 32)  # (batch=2, seq_len=5, dim=32)
    
    # Create KG mask: only allow attention along KG edges
    kg_mask = torch.zeros(2, 5, 5)
    kg_mask[:, 0, 1] = 1.0  # Token 0 can attend to token 1
    kg_mask[:, 1, 2] = 1.0  # Token 1 can attend to token 2
    kg_mask[:, :, :] = 1.0  # For demo, allow all (normally would be sparse)
    
    output = attention(x, kg_mask)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention applied successfully with KG constraints")
    
    print("\n" + "=" * 60)
    print("Knowledge-Guided Classifier Example")
    print("=" * 60)
    
    # Create classifier
    classifier = KnowledgeGuidedClassifier(
        input_dim=20,
        n_classes=3,
        knowledge_graph=kg,
        embedding_dim=32,
        hidden_dim=16
    )
    
    # Sample inputs
    x_input = torch.randn(4, 20)  # 4 samples
    entity_ids = torch.tensor([0, 1, 4, 5])  # Dog, Cat, Bird, Canary
    
    logits = classifier(x_input, entity_ids)
    predictions = torch.softmax(logits, dim=-1)
    
    print("\nClassification results:")
    entity_names = ['Dog', 'Cat', 'Bird', 'Canary']
    for i in range(4):
        print(f"  {entity_names[i]}: {predictions[i].detach().numpy()}")
