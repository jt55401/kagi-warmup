#!/bin/bash

data=$(cat <<EOF
{
  "profile": "I am a theoretical biologist, interested in disease ecology. My tools are R, clojure , compartmentalism disease modeling, and statistical GAM models, using a variety of data layers (geophysical, reconstructions, climate, biodiversity, land use). Besides that I am interested in tech applied to the a subset of the current problems of the world (agriculture / biodiversity / conservation / forecasting), development of third world countries and AI, large language models.",
  "number": 10
}
EOF
)

curl --location --request POST 'http://localhost:5000/news' \
--header 'Content-Type: application/json' \
--data-raw "$data"