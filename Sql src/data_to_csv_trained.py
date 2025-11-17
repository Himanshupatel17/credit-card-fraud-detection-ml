"""
X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)
y_train_df = pd.DataFrame(y_train, columns=["Class"])
y_test_df = pd.DataFrame(y_test, columns=["Class"])

# Save to processed folder
X_train_df.to_csv("D:\domain projects\credit card\dataset\Processed/X_train.csv", index=False)
X_test_df.to_csv("D:\domain projects\credit card\dataset\Processed/X_test.csv", index=False)
y_train_df.to_csv("D:\domain projects\credit card\dataset\Processed/y_train.csv", index=False)
y_test_df.to_csv("D:\domain projects\credit card\dataset\Processed/y_test.csv", index=False)

print(" Preprocessed train/test datasets saved ")
"""