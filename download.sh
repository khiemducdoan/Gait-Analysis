#!/bin/bash
set -e

API_TOKEN="997ef826-53b2-43d5-83d3-66a23a08487c"                          # ← Replace with your real token
SERVER_URL="https://borealisdata.ca"        # ← Crucial: NOT demo.dataverse.org
PERSISTENT_ID="doi:10.5683/SP3/TWIKMK"

# Step 1: Fetch dataset metadata to get file IDs
echo "Fetching dataset metadata..."
metadata=$(curl -s -H "X-Dataverse-key: $API_TOKEN" \
  "$SERVER_URL/api/datasets/:persistentId/?persistentId=$PERSISTENT_ID")

# Step 2: Extract file IDs and filenames using jq (install jq if needed)
if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' is required to parse JSON. Install it with 'sudo apt install jq' or equivalent."
    exit 1
fi

echo "Files in dataset:"
echo "$metadata" | jq -r '.data.latestVersion.files[] | "\(.dataFile.id) \(.dataFile.filename)"'

# Step 3: Download each file
echo "$metadata" | jq -r '.data.latestVersion.files[] | "\(.dataFile.id) \(.dataFile.filename)"' | while read -r fileid filename; do
    echo "Downloading $filename (ID: $fileid)..."
    curl -L -H "X-Dataverse-key: $API_TOKEN" \
         -o "assets/datasets/${filename}" \
         "$SERVER_URL/api/access/datafile/$fileid"
done

echo "✅ Download complete!"