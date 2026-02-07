"""
View Blockchain Audit Trail
"""

from blockchain import BlockchainManager

def view_audit():
    """Display complete blockchain audit trail"""
    
    print("\n" + "=" * 60)
    print("CONNECTING TO BLOCKCHAIN")
    print("=" * 60)
    
    bm = BlockchainManager()
    
    # Print full audit trail
    bm.print_audit_trail()
    
    # Additional statistics
    total = bm.get_total_updates()
    
    print("=" * 60)
    print("STATISTICS BY CLIENT")
    print("=" * 60)
    
    for client_id in range(1, 4):
        client_updates = bm.get_client_updates(client_id)
        print(f"\nClient {client_id}:")
        print(f"  Total contributions: {len(client_updates)}")
        
        # Get accuracies
        accs = []
        for idx in client_updates:
            update = bm.get_update_details(idx)
            accs.append(update['accuracy'])
        
        if accs:
            print(f"  Average accuracy: {sum(accs)/len(accs):.4f}")
            print(f"  Best accuracy: {max(accs):.4f}")
            print(f"  Latest accuracy: {accs[-1]:.4f}")
    
    print("\n" + "=" * 60)
    print("STATISTICS BY ROUND")
    print("=" * 60)
    
    for round_num in range(1, 21):
        round_updates = bm.get_round_updates(round_num)
        
        if len(round_updates) > 0:
            print(f"\nRound {round_num}:")
            print(f"  Participants: {len(round_updates)} clients")
            
            accs = []
            for idx in round_updates:
                update = bm.get_update_details(idx)
                accs.append(update['accuracy'])
            
            print(f"  Average accuracy: {sum(accs)/len(accs):.4f}")

if __name__ == "__main__":
    view_audit()


