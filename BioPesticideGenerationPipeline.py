# Welcome to the repository everyone! I will solve the problem of potent pesticides by generating sequences to be used in farming. These will have less effect on food, but also increase resistance.
# Following are necessary imports for sequence generation + running 3 CNN models to understand motifs.

import os
import torch 
import random 
import torch.nn as nn  
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier 


#The following data parser ensures that all FASTA data is able to get enumarted (creating a vocabulary table) which assigns an index for each amino acid allowing for teknization later on#
def parse_fasta_folder(folder_path):
    sequences = []
    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' not found. Check your working directory.")
        return sequences
        
    for filename in os.listdir(folder_path):
        if filename.endswith(".fasta"):
            file_path = os.path.join(folder_path, filename)
            current_seq = []
            
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(">"):
                        if current_seq:
                            seq_str = "".join(current_seq).upper()
                            if all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq_str):
                                sequences.append(seq_str)
                            current_seq = []
                    else:
                        current_seq.append(line)
            
            if current_seq:
                seq_str = "".join(current_seq).upper()
                if all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq_str):
                    sequences.append(seq_str)
                    
    return list(set(sequences))


# This gives a shared vocabulary which assigns each a number with first letter beginning at 0
VOCAB = {char: idx + 1 for idx, char in enumerate("ACDEFGHIKLMNPQRSTVWY")} #The "VOCAB" sicn eit is a fixed variable and lets teh computer know that no matter what the variables within this cannot be changed no matter the case#
VOCAB['<PAD>'] = 0  

# The following class gives the CNN architecture / building blocks of how code will generate
class MotifDataset(Dataset):
    def __init__(self, positive_sequences, vocab, max_length=64):
        self.vocab = vocab
        self.max_len = max_length  # Fixed reference here
        self.samples = []
        self.labels = []

        # I will first add a "positive sample" sequence parsing function which are sequences found within my training dataset 
        for seq in positive_sequences:
            self.samples.append(self.tokenize(seq))
            self.labels.append(1.0) # If the following for loop produces one this ensures that it is a real sequences. This labels it with the number 1

        # Now here is the catch, I will create a for loop where it shuffles the sequences within my training data to not only reduce the risk of memorization, but also aids the CNN in understanding what is wrong.
        for seq in positive_sequences:
            shuffled_seq = "".join(random.sample(seq, len(seq)))
            self.samples.append(self.tokenize(shuffled_seq))
            self.labels.append(0.0)

    def tokenize(self, seq): 
        # The tokenization occurs here where each amino acid is given a number and vectorized
        encoded = [self.vocab.get(c, 0) for c in seq]
        if len(encoded) < self.max_len:
            encoded += [0] * (self.max_len - len(encoded))
        return encoded[:self.max_len]
    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)
    


# I will now create the code for the training loop for the CNN with proper channels, kernels, etc. RELU (Rectified Linear Unit) will be implemented to ensure no dead neurons due to negative values within weighting vector killing the backpropagation.
class MotifCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=3, padding=1) #This 1D convulutional netwoek has a 3x1 kernel vector 
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, 1)
        
    def forward(self, x, return_embedding=False):
        x = self.embedding(x).transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        pooled = self.pool(x).squeeze(-1)
        
        if return_embedding:
            return pooled # Returns the 128-dim motif embedding
        return self.fc(pooled)

def train_folder_cnn(cnn_model, sequences, folder_name, epochs=15, batch_size=32):
    """The following function trains a CNN on a specific folder's data and returns the frozen model."""
    print(f"\nTraining Motif CNN for {folder_name}")
    
    if len(sequences) == 0:
        print(f"Skipping {folder_name}: No sequences provided.")
        return cnn_model

    dataset = MotifDataset(sequences, VOCAB)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Binary Cross Entropy with Logits for 1 vs 0 classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=1e-3)
    
    cnn_model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = cnn_model(inputs).squeeze(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
        
    cnn_model.eval() # Freeze the model for inference and to input into Transformer Later on#
    print(f"{folder_name} CNN successfully trained and frozen.")
    return cnn_model


#---------Tranfformer Architecture + training-----------#
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class PeptideTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=3, max_len=64): #The transformer has 4 attention heads and 3 linear layers. Mainly because I am running this on CPU, but because more does not mean better#
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)


#I am now training the transformer#
class TransformerDataset(Dataset):
    def __init__(self, sequences, vocab, max_len=64):
        self.vocab = vocab
        self.max_len = max_len
        self.encoded_seqs = []
        
        for seq in sequences:
            # Shifted encoding for autoregressive prediction
            encoded = [self.vocab.get(c, 0) for c in seq]
            if len(encoded) < max_len:
                encoded += [0] * (max_len - len(encoded))
            self.encoded_seqs.append(encoded[:max_len])
            
    def __len__(self):
        return len(self.encoded_seqs)
        
    def __getitem__(self, idx):
        seq = torch.tensor(self.encoded_seqs[idx], dtype=torch.long)
        # Inputs are everything except the last token; Targets are everything except the first
        return seq[:-1], seq[1:]

def train_transformer(model, sequences, epochs=20, batch_size=32):
    print("\n--- Training Generative Transformer ---")
    dataset = TransformerDataset(sequences, VOCAB)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore padding token
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm = 1.0)

            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1:02d}/{epochs} | Generative Loss: {total_loss/len(dataloader):.4f}")
    
    model.eval()
    print("Transformer successfully trained.")
    return model

#After this since the transformer has been trained, I will now perform the sequence genration#
def generate_pesticide_candidate(model, vocab, max_generate_len=50, temperature=0.85):
    """Autoregressively generates a novel pesticide sequence from the trained Transformer."""
    model.eval()
    inverse_vocab = {v: k for k, v in vocab.items()}
    
    with torch.no_grad():
        idx = torch.tensor([[random.choice(list(vocab.values()))]], dtype=torch.long)
        
        for _ in range(max_generate_len):
            idx_cond = idx[:, -64:]
            logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            
            if next_token.item() == 0: 
                break
                
        generated_indices = idx.squeeze().tolist()
        if not isinstance(generated_indices, list):
            generated_indices = [generated_indices]
        generated_letters = [inverse_vocab.get(i, '') for i in generated_indices if i != 0]
        return "".join(generated_letters)


def extract_rf_features(sequence):
    """Extracts amino acid frequencies, total length, and approximate charge balance. The random forst model allows for the chemical properties be placed in tabular format."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    counts = [sequence.count(aa) / max(1, len(sequence)) for aa in amino_acids]
    net_charge = (sequence.count('K') + sequence.count('R')) - (sequence.count('D') + sequence.count('E'))
    return counts + [len(sequence), net_charge]


def run_integrated_discovery_pipeline(transformer_model, cnn_amp, cnn_data, cnn_more, rf_model, vocab, num_candidates=3):
    """The following function brings all 3 CNN models + Transformer which are frozemn to cretae new sequences."""
    print(f"\nStarting discovery screening...")
    
    successful_discoveries = []
    cnn_amp.eval()
    cnn_data.eval()
    cnn_more.eval()
    
    attempts = 0
    while len(successful_discoveries) < num_candidates and attempts < 200:
        attempts += 1
        candidate = generate_pesticide_candidate(transformer_model, vocab)
        
        if len(candidate) < 10:
            continue
            
        # Format candidate tensor for the 3 CNN evaluators
        encoded_seq = [vocab.get(c, 0) for c in candidate]
        if len(encoded_seq) < 64:
            encoded_seq += [0] * (64 - len(encoded_seq))
        cnn_input = torch.tensor([encoded_seq[:64]], dtype=torch.long)
        
        # Pull functional motif scores. These are the weights generated from each run for the ranking. Rmeber, sigmoids peak at 1 and have a minimum at 0.
        # The use of sigmoid allows for gradient vectors not to dissapear within a neural network.#
        with torch.no_grad():
            score_amp = torch.sigmoid(cnn_amp(cnn_input)).item()
            score_data = torch.sigmoid(cnn_data(cnn_input)).item()
            score_more = torch.sigmoid(cnn_more(cnn_input)).item()
            
        # Run through physical parameter check
        rf_features = [extract_rf_features(candidate)]
        rf_pass = rf_model.predict(rf_features)[0]
        rf_prob = rf_model.predict_proba(rf_features)[0][1]
        
        # Compute the global system fitness index
        viability_index = (score_amp + score_data + score_more + rf_prob) / 4
        
        # Screen candidates based on structural integrity thresholds
        if rf_pass == 1.0 and viability_index > 0.65:
            successful_discoveries.append({
                "sequence": candidate,
                "viability": viability_index
            })
            
            print(f"  SUCCESSFUL DISCOVERY #{len(successful_discoveries)}: {candidate}")
            print(f"   AMPDataset Motif: {score_amp:.4f} | AntimicrobialData Motif: {score_data:.4f}")
            print(f"   AntimicrobialMore Motif: {score_more:.4f} | RF Attributes Prob: {rf_prob:.4f}")
            print(f"   INTEGRATED SYSTEM VIABILITY INDEX: {viability_index:.4f}\n")

    return successful_discoveries

if __name__ == "__main__": #This ensures that all the draining happens under the project main tab to ensure it works. I haveeven added the transformer module.
    amp_dataset_seqs = parse_fasta_folder("AMPDataset")
    anti_data_seqs = parse_fasta_folder("AntimicrobialData")
    anti_more_seqs = parse_fasta_folder("AntimicrobialMore")

    cnn_amp = MotifCNN(vocab_size=len(VOCAB) + 1)
    cnn_data = MotifCNN(vocab_size=len(VOCAB) + 1)
    cnn_more = MotifCNN(vocab_size=len(VOCAB) + 1)

    cnn_amp = train_folder_cnn(cnn_amp, amp_dataset_seqs, "AMPDataset")
    cnn_data = train_folder_cnn(cnn_data, anti_data_seqs, "AntimicrobialData")
    cnn_more = train_folder_cnn(cnn_more, anti_more_seqs, "AntimicrobialMore")

    
    all_transformer_seqs = list(set(amp_dataset_seqs + anti_data_seqs + anti_more_seqs))
    
    generator_model = PeptideTransformer(vocab_size=len(VOCAB))
    generator_model = train_transformer(generator_model, all_transformer_seqs, epochs=20)

    #Training the Random Forest on real sequence properties
    print("\nTraining Random Forest Model...")
    X_rf = []
    y_rf = []

    for seq in all_transformer_seqs:
        X_rf.append(extract_rf_features(seq))
        y_rf.append(1.0)
        
    for seq in all_transformer_seqs:
        shuffled = "".join(random.sample(seq, len(seq)))
        X_rf.append(extract_rf_features(shuffled))
        y_rf.append(0.0)
    
    rf_gatekeeper = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_gatekeeper.fit(X_rf, y_rf)
    print("Random Forest trained successfully.")

    final_pesticides = run_integrated_discovery_pipeline(
        generator_model, 
        cnn_amp, 
        cnn_data, 
        cnn_more, 
        rf_gatekeeper, 
        VOCAB, 
        num_candidates=3
    )



#The score scalign is between 1 and 0 with "1" representing the best result and "0" respresenting the worsdt sequence#
#During first iteration of pipeline, teh follwoing are the reults: 


# SUCCESSFUL DISCOVERY #1: NWFRKWKWWRKKWFKVAFKWWVKKRVWRKRFWHAIRGRKKRGKKVGYYLSQ 
#   AMPDataset Motif: 1.0000 | AntimicrobialData Motif: 0.9985
#   AntimicrobialMore Motif: 0.0970 | RF Attributes Prob: 0.5284
#   INTEGRATED SYSTEM VIABILITY INDEX: 0.6560

#  SUCCESSFUL DISCOVERY #2: VIPYNANLAARTVARNLAGEAANTTVWRSCETMRNNAASSVCSSNLNKSL
#  AMPDataset Motif: 0.6666 | AntimicrobialData Motif: 0.7103
#  AntimicrobialMore Motif: 0.9983 | RF Attributes Prob: 0.5095
#  INTEGRATED SYSTEM VIABILITY INDEX: 0.7212

#  SUCCESSFUL DISCOVERY #3: DKLIGSCVWGATNYTSRCNAECKRRGYKGGHCGSFANVNCWCETVCTYPNI
#  AMPDataset Motif: 1.0000 | AntimicrobialData Motif: 0.9983
#  AntimicrobialMore Motif: 0.9952 | RF Attributes Prob: 0.5331
#  INTEGRATED SYSTEM VIABILITY INDEX: 0.8817


#To further vlaidate these sequences. I ran the sequences through alphafold webserver#

