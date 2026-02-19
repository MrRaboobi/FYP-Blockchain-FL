"""
test_oracle.py — Phase 1.3 Oracle Verification Script

Programmatically:
  1. Deploy a fresh FLLogger contract.
  2. Trigger a dummy logUpdate transaction (emits ModelUpdate event).
  3. Call fetch_client_history() and verify the log is retrieved correctly.
"""

import sys
import json
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from web3 import Web3
from solcx import compile_source, install_solc

from blockchain import fetch_client_history


GANACHE_URL = 'http://127.0.0.1:8545'
SOL_VERSION = '0.8.19'
CONTRACT_PATH = Path('contracts/FLLogger.sol')


def compile_contract():
    install_solc(SOL_VERSION, show_progress=False)
    source = CONTRACT_PATH.read_text()
    compiled = compile_source(source, output_values=['abi', 'bin'], solc_version=SOL_VERSION)
    _, interface = compiled.popitem()
    return interface


def deploy_fresh_contract(w3, interface):
    deployer = w3.eth.accounts[0]
    Contract = w3.eth.contract(
        abi=interface['abi'],
        bytecode=interface['bin']
    )
    tx_hash = Contract.constructor().transact({'from': deployer})
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    contract = w3.eth.contract(address=receipt.contractAddress, abi=interface['abi'])
    return contract, deployer


def main():
    print("=" * 60)
    print("ORACLE TEST — fetch_client_history")
    print("=" * 60)

    # Connect
    w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
    assert w3.is_connected(), "Cannot connect to Ganache on " + GANACHE_URL
    print(f"[OK] Connected to Ganache (block={w3.eth.block_number})")

    # Compile + deploy fresh contract so history is clean
    print("[..] Compiling FLLogger.sol ...")
    interface = compile_contract()
    print("[OK] Compilation successful")

    print("[..] Deploying fresh FLLogger contract ...")
    contract, deployer = deploy_fresh_contract(w3, interface)
    print(f"[OK] Deployed at {contract.address}")

    # Trigger dummy logUpdate — msg.sender will be the deployer address
    dummy_round    = 7
    dummy_accuracy = 8750   # represents 87.50%
    dummy_hash     = b'\xab' * 32
    dummy_datasize = 500

    print(f"[..] Submitting dummy logUpdate (round={dummy_round}, accuracy_int={dummy_accuracy}) ...")
    tx_hash = contract.functions.logUpdate(
        dummy_round,
        1,              # clientId
        dummy_hash,
        dummy_datasize,
        dummy_accuracy
    ).transact({'from': deployer})
    w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"[OK] Transaction confirmed: {tx_hash.hex()}")

    # Call fetch_client_history for the deployer (who was msg.sender)
    print(f"[..] Calling fetch_client_history for deployer address: {deployer} ...")
    history = fetch_client_history(deployer, contract, w3)

    # Validate
    assert len(history) == 1, f"Expected 1 log entry, got {len(history)}"
    entry = history[0]

    assert entry['round'] == dummy_round, \
        f"Round mismatch: expected {dummy_round}, got {entry['round']}"
    assert abs(entry['accuracy'] - dummy_accuracy / 10000.0) < 1e-9, \
        f"Accuracy mismatch: expected {dummy_accuracy/10000.0}, got {entry['accuracy']}"
    assert isinstance(entry['timestamp'], int) and entry['timestamp'] > 0, \
        "Timestamp must be a positive integer"

    print()
    print("=" * 60)
    print("RETRIEVED LOG ENTRY:")
    print(f"  round     : {entry['round']}")
    print(f"  accuracy  : {entry['accuracy']:.4f}")
    print(f"  timestamp : {entry['timestamp']}")
    print("=" * 60)
    print("[PASS] fetch_client_history verified successfully.")


if __name__ == '__main__':
    main()
