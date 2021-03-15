# Script used to retrieve all CORE documents with language label Russian,
# Ukrainian, Bulgarian, or Macedonian

for f in ./*.xz; do
    echo $f
    unxz --to-stdout $f | grep -P "\"language\": \{\"code\": \"((ru)|(uk)|(bg)|(mk))\"" >> core_all_cyr.jsonl
done
