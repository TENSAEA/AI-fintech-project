import pandas as pd
import numpy as np
import networkx as nx
import random
from datetime import datetime, timedelta
import os

def generate_synthetic_data(n_transactions=10000, n_accounts=1000, fraud_ratio=0.02):
    """
    Generates synthetic transaction data with fraud patterns.
    """
    print(f"Generating {n_transactions} transactions for {n_accounts} accounts...")
    
    # Accounts
    account_ids = [f"ACC_{i}" for i in range(n_accounts)]
    
    # Time range (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data = []
    
    # 1. Generate Normal Transactions
    n_fraud = int(n_transactions * fraud_ratio)
    n_normal = n_transactions - n_fraud
    
    print(f"Generating {n_normal} normal transactions...")
    for _ in range(n_normal):
        sender = random.choice(account_ids)
        receiver = random.choice(account_ids)
        while receiver == sender:
            receiver = random.choice(account_ids)
            
        amount = round(np.random.lognormal(mean=3, sigma=1), 2) # Lognormal distribution for amounts
        timestamp = start_date + timedelta(seconds=random.randint(0, 30*24*3600))
        
        data.append({
            "transaction_id": f"TXN_{len(data)}",
            "sender_id": sender,
            "receiver_id": receiver,
            "amount": amount,
            "timestamp": timestamp,
            "is_fraud": 0,
            "type": "TRANSFER"
        })

    # 2. Generate Fraud Rings (Graph Pattern)
    print(f"Generating {n_fraud} fraud transactions (including rings)...")
    # Create a few fraud rings
    n_rings = 5
    ring_size = 10
    fraud_accounts = random.sample(account_ids, n_rings * ring_size)
    
    for i in range(n_rings):
        ring_accounts = fraud_accounts[i*ring_size : (i+1)*ring_size]
        # Cycle transaction: A -> B -> C -> ... -> A
        for j in range(len(ring_accounts)):
            sender = ring_accounts[j]
            receiver = ring_accounts[(j+1) % len(ring_accounts)]
            
            amount = round(np.random.uniform(5000, 10000), 2) # High amounts for fraud
            # Fraud often happens in quick succession or odd hours, let's randomize within a short window
            base_time = start_date + timedelta(days=random.randint(0, 28))
            timestamp = base_time + timedelta(minutes=random.randint(0, 60))
            
            data.append({
                "transaction_id": f"TXN_{len(data)}",
                "sender_id": sender,
                "receiver_id": receiver,
                "amount": amount,
                "timestamp": timestamp,
                "is_fraud": 1,
                "type": "TRANSFER"
            })
            
    # Fill remaining fraud with random high value or rapid transactions
    remaining_fraud = n_fraud - len(data) + n_normal
    if remaining_fraud > 0:
        for _ in range(remaining_fraud):
            sender = random.choice(account_ids)
            receiver = random.choice(account_ids)
            amount = round(np.random.uniform(1000, 5000), 2)
            timestamp = start_date + timedelta(seconds=random.randint(0, 30*24*3600))
            
            data.append({
                "transaction_id": f"TXN_{len(data)}",
                "sender_id": sender,
                "receiver_id": receiver,
                "amount": amount,
                "timestamp": timestamp,
                "is_fraud": 1,
                "type": "TRANSFER"
            })

    df = pd.DataFrame(data)
    
    # Sort by time
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Save to CSV
    output_path = "data/transactions.csv"
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    return df

if __name__ == "__main__":
    generate_synthetic_data()
