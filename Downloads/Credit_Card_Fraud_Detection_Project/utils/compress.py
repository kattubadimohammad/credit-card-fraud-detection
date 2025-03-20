import zipfile
import os

def compress_csv(input_file="data/creditcard.csv", output_file="data/creditcard.zip"):
    """Compress CSV file into a ZIP archive."""
    if not os.path.exists(input_file):
        print(f"⚠️ File {input_file} not found!")
        return
    
    with zipfile.ZipFile(output_file, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(input_file, arcname="creditcard.csv")
    
    print(f"✅ CSV file successfully compressed to {output_file}")

if __name__ == "__main__":
    compress_csv()
