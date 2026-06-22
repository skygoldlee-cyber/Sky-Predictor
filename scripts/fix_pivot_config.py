import json

with open('config.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

# Add "confirmed" to events if not present
events = d['adaptive_indicator']['pivot_candidate_alert']['events']
if 'confirmed' not in events:
    events.append('confirmed')
    print("Added 'confirmed' to events")
else:
    print("'confirmed' already in events")

with open('config.json', 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)

print("Config updated successfully")
