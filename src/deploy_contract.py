"""
Deploy FLLogger Smart Contract to Ganache
"""

from web3 import Web3
from solcx import compile_source
from pathlib import Path
import json

def compile_contract():
    """Compile the Solidity contract"""
    
    print("=" * 60)
    print("Compiling Smart Contract")
    print("=" * 60)
    
    # Read contract source
    contract_path = Path('contracts/FLLogger.sol')
    
    with open(contract_path, 'r') as f:
        contract_source = f.read()
    
    # Compile
    print("\nCompiling FLLogger.sol...")
    compiled_sol = compile_source(
        contract_source,
        output_values=['abi', 'bin'],
        solc_version='0.8.19'
    )
    
    # Get contract interface
    contract_id, contract_interface = compiled_sol.popitem()
    
    print("✅ Contract compiled successfully!")
    
    return contract_interface

def deploy_contract():
    """Deploy contract to Ganache"""
    
    print("\n" + "=" * 60)
    print("Deploying to Ganache")
    print("=" * 60)
    
    # Connect to Ganache
    w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
    
    if not w3.is_connected():
        print("❌ Cannot connect to Ganache!")
        print("Make sure Ganache is running on port 8545")
        return None
    
    print(f"\n✅ Connected to Ganache")
    print(f"Block number: {w3.eth.block_number}")
    
    # Compile contract
    contract_interface = compile_contract()
    
    # Get deployment account
    deployer = w3.eth.accounts[0]
    print(f"\nDeploying from account: {deployer}")
    
    # Create contract instance
    FLLogger = w3.eth.contract(
        abi=contract_interface['abi'],
        bytecode=contract_interface['bin']
    )
    
    # Deploy
    print("\nDeploying contract...")
    tx_hash = FLLogger.constructor().transact({'from': deployer})
    
    # Wait for transaction receipt
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    contract_address = tx_receipt.contractAddress
    
    print(f"✅ Contract deployed at: {contract_address}")
    print(f"Gas used: {tx_receipt.gasUsed:,}")
    
    # Save contract info
    contract_info = {
        'address': contract_address,
        'abi': contract_interface['abi'],
        'deployer': deployer
    }
    
    output_path = Path('contracts/deployed_contract.json')
    with open(output_path, 'w') as f:
        json.dump(contract_info, f, indent=2)
    
    print(f"\n✅ Contract info saved to: {output_path}")
    
    # Test the contract
    print("\n" + "=" * 60)
    print("Testing Contract")
    print("=" * 60)
    
    contract = w3.eth.contract(
        address=contract_address,
        abi=contract_interface['abi']
    )
    
    # Test: Log a dummy update
    print("\nLogging test update...")
    tx_hash = contract.functions.logUpdate(
        1,      # round
        1,      # clientId
        b'\x00' * 32,  # modelHash (dummy)
        1000,   # dataSize
        9580    # accuracy (95.80%)
    ).transact({'from': deployer})
    
    w3.eth.wait_for_transaction_receipt(tx_hash)
    
    # Read back
    total_updates = contract.functions.getTotalUpdates().call()
    print(f"✅ Total updates: {total_updates}")
    
    update = contract.functions.getUpdate(0).call()
    print(f"✅ Test update retrieved:")
    print(f"   Round: {update[0]}")
    print(f"   Client: {update[1]}")
    print(f"   Accuracy: {update[5] / 100:.2f}%")
    
    print("\n" + "=" * 60)
    print("🎉 DEPLOYMENT COMPLETE!")
    print("=" * 60)
    print(f"\nContract Address: {contract_address}")
    print(f"Save this address - you'll need it for FedAvg integration!")
    
    return contract_address

if __name__ == "__main__":
    deploy_contract()

