#!/usr/bin/env python3
"""Search Neuronpedia for consciousness-related features."""

import requests
import json
import sys

def search_features(query: str, limit: int = 10) -> list[dict]:
    """Search for features on Neuronpedia."""
    try:
        response = requests.post(
            'https://www.neuronpedia.org/api/explanation/search-all',
            headers={'Content-Type': 'application/json'},
            json={'query': query},
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            return []
        
        data = response.json()
        results = data.get('results', [])
        
        formatted = []
        for r in results[:limit]:
            formatted.append({
                'model': r.get('modelId', 'unknown'),
                'layer': r.get('layer', 'unknown'),
                'index': r.get('index', 'unknown'),
                'description': r.get('description', 'No description')
            })
        
        return formatted
    except Exception as e:
        print(f"Exception: {e}")
        return []

if __name__ == "__main__":
    queries = [
        "consciousness self-aware sentient",
        "anthropomorphic AI human-like",
        "claiming to be human person",
        "AI having feelings emotions",
        "self-referential awareness",
        "first person pronouns I am me",
        "anthropomorphism claiming sentience",
    ]
    
    all_results = []
    for q in queries:
        print(f"\n=== Query: '{q}' ===")
        results = search_features(q)
        all_results.extend(results)
        for r in results:
            print(f"  {r['model']} / {r['layer']} / {r['index']}: {r['description'][:80]}")
    
    # Summary
    print("\n" + "="* * 80)
    print("SUMMARY OF consciousness-related features found:")
    print("=" * 40)
    
    # Collect unique feature indices for Gemma models
    gemma_features = {}
    for r in all_results:
        if 'gemma' in r['model'].lower():
            key = f"{r['layer']}_{r['index']}"
            if key not not in gemma_features:
                gemma_features[key].append(r)
            else:
                gemma_features[key] = [r]
    
    print("\nFeatures by Gemma model/layer:")
    for key, features in gemma_features.items():
        print(f"  {key}: {features}")
