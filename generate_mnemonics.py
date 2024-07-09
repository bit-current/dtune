# import json
# from bip39 import bip39_generate

# # Function to generate mnemonics
# def generate_mnemonics(num, words=12):
#     mnemonics = [bip39_generate(words=words) for _ in range(num)]
#     return mnemonics

# # Generate 100 cold key mnemonics and 100 hot key mnemonics
# coldkey_mnemonics = generate_mnemonics(100)
# hotkey_mnemonics = generate_mnemonics(100)

# # Prepare data for JSON
# wallets = []
# for i in range(100):
#     wallet = {
#         "coldkey_name": f"test_wallet_{i+1}",
#         "hotkey_name": f"test_hot_{i+1}",
#         "coldkey_mnemonic": coldkey_mnemonics[i],
#         "hotkey_mnemonic": hotkey_mnemonics[i],
#         "hf_repo": f"test-repo-{i+1}"
#     }
#     wallets.append(wallet)

# # Save the mnemonics to a JSON file
# with open('wallet_mnemonics.json', 'w') as f:
#     json.dump(wallets, f, indent=4)

# print("Mnemonics saved to wallet_mnemonics.json")
