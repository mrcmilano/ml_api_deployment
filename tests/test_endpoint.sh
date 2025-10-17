#!/bin/bash

# URL base dell'app
BASE_URL="http://127.0.0.1:8000"

echo "🔍 Test: /status"
curl -s "$BASE_URL/status" | jq
echo "\n"

echo "🔍 Test: /model_version"
curl -s "$BASE_URL/model_version" | jq
echo "\n"

echo "🔍 Test: /language_detection with 4 languages"
curl -s -X POST "$BASE_URL/language_detection" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Hello world", "Bonjour le monde", "Ciao mondo", "Hola amigo"]}' | jq
echo "\n"

echo "🔍 Test: /language_detection with unknown language"
curl -s -X POST "$BASE_URL/language_detection" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Dum loquimur fugit invida aetas"]}' | jq
echo "\n"

echo "🔍 Test: /language_detection with 1 element in texts"
curl -s -X POST "$BASE_URL/language_detection" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["proviamo con un solo elemento nella lista"]}' | jq
echo "\n"

echo "🔍 Test: /language_detection with empty input"
curl -s -X POST "$BASE_URL/language_detection" \
     -H "Content-Type: application/json" \
     -d '{"texts": []}' | jq
echo "\n"

# TODO:
# echo "🔍 Test: /language_detection with invalid input"
# curl -s -X POST "$BASE_URL/language_detection" \
#      -H "Content-Type: application/json" \
#      -d '{"text": "This is wrong"}' | jq
# echo -e "\n"