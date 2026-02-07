"""
Blockchain Manager for FL Provenance Logging
"""

from web3 import Web3
import json
from pathlib import Path
import hashlib
import pickle

class BlockchainManager:
    """Manages blockchain interactions for federated learning"""
    
    def __init__(self, ganache_url='http://127.0.0.1:8545'):
        """Initialize connection to Ganache"""
        
        self.w3 = Web3(Web3.HTTPProvider(ganache_url))
        
        if not self.w3.is_connected():
            raise ConnectionError("Cannot connect to Ganache blockchain")
        
        # Load deployed contract info
        contract_info_path = Path('contracts/deployed_contract.json')
        
        with open(contract_info_path, 'r') as f:
            contract_info = json.load(f)
        
        self.contract_address = contract_info['address']
        self.contract_abi = contract_info['abi']
        self.deployer = contract_info['deployer']
        
        # Create contract instance
        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=self.contract_abi
        )
        
        print(f"✅ Connected to blockchain")
        print(f"   Contract: {self.contract_address}")
        print(f"   Block: {self.w3.eth.block_number}")
    
    def hash_model(self, model_state_dict):
        """
        Create SHA-256 hash of model weights
        
        Args:
            model_state_dict: PyTorch model.state_dict()
        
        Returns:
            bytes32 hash
        """
        # Serialize model weights
        model_bytes = pickle.dumps(model_state_dict)
        
        # Compute SHA-256
        hash_hex = hashlib.sha256(model_bytes).hexdigest()
        
        # Convert to bytes32 for Solidity
        hash_bytes = bytes.fromhex(hash_hex)
        
        return hash_bytes
    
    def log_update(self, round_num, client_id, model_state_dict, data_size, accuracy):
        """
        Log a model update on-chain
        
        Args:
            round_num: Training round number
            client_id: Client identifier (1, 2, 3)
            model_state_dict: PyTorch model.state_dict()
            data_size: Number of training samples
            accuracy: Validation accuracy (0-1 float)
        
        Returns:
            Transaction receipt
        """
        # Hash the model
        model_hash = self.hash_model(model_state_dict)
        
        # Convert accuracy to integer (e.g., 0.9580 -> 9580)
        accuracy_int = int(accuracy * 10000)
        
        # Call smart contract
        tx_hash = self.contract.functions.logUpdate(
            round_num,
            client_id,
            model_hash,
            data_size,
            accuracy_int
        ).transact({'from': self.deployer})
        
        # Wait for transaction
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return receipt
    
    def complete_round(self, round_num):
        """Mark a round as completed"""
        
        tx_hash = self.contract.functions.completeRound(
            round_num
        ).transact({'from': self.deployer})
        
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return receipt
    
    def get_total_updates(self):
        """Get total number of logged updates"""
        return self.contract.functions.getTotalUpdates().call()
    
    def get_round_updates(self, round_num):
        """Get all updates for a specific round"""
        return self.contract.functions.getRoundUpdates(round_num).call()
    
    def get_client_updates(self, client_id):
        """Get all updates from a specific client"""
        return self.contract.functions.getClientUpdates(client_id).call()
    
    def get_update_details(self, update_index):
        """Get details of a specific update"""
        update = self.contract.functions.getUpdate(update_index).call()
        
        return {
            'round': update[0],
            'client_id': update[1],
            'model_hash': update[2].hex(),
            'data_size': update[3],
            'timestamp': update[4],
            'accuracy': update[5] / 10000  # Convert back to float
        }
    
    def print_audit_trail(self):
        """Print complete audit trail"""
        
        total = self.get_total_updates()
        
        print("\n" + "=" * 60)
        print("BLOCKCHAIN AUDIT TRAIL")
        print("=" * 60)
        print(f"Total updates logged: {total}\n")
        
        for i in range(total):
            update = self.get_update_details(i)
            print(f"Update #{i}:")
            print(f"  Round: {update['round']}")
            print(f"  Client: {update['client_id']}")
            print(f"  Accuracy: {update['accuracy']:.4f}")
            print(f"  Data size: {update['data_size']}")
            print(f"  Hash: {update['model_hash'][:16]}...")
            print()

# Test
if __name__ == "__main__":
    bm = BlockchainManager()
    bm.print_audit_trail()


