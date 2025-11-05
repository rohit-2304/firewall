# Transformer-based Web Application Firewall (WAF)

## Overview

A proof-of-concept implementation of a transformer-driven, machine learning-powered Web Application Firewall (WAF) designed for real-time detection of malicious web traffic in modern environments. Integrates advanced models such as BERT, LogBERT, Longformer, and Graph Neural Networks for identifying both common and zero-day web attacks, with deployment-ready containerization for seamless infrastructure integration[attached_file:1][file:2].

## Features

- Transformer-based threat detection (BERT/LogBERT/Longformer).
- Graph-based behavioral analysis with GNNs for detecting stealthy, multi-step attacks.
- Mixture of Experts architecture for dynamic model selection.
- Online and federated learning for continuous adaptation.
- Interactive traffic dashboard and real-time analytics.
- Plug-and-play deployment using Docker and microservices.
- Low-latency, scalable, async architecture with GPU acceleration.

## Quickstart

1. **Clone OWASP Juice Shop (for testing):**
    ```
    git clone https://github.com/juice-shop/juice-shop
    ```

2. **Build Juice Shop Docker Image:**
    ```
    cd juice-shop
    docker build -t juice-shop .
    ```

3. **Download or train your detection model:**
    - Place model files (e.g., `.pt`, `.bin`) in the `models/` directory.

4. **Build and run the firewall containers:**
    ```
    git clone https://github.com/rohit-2304/firewall
    cd firewall
    docker-compose up -d
    ```

## Architecture

- Reverse proxy forwards HTTP requests to the firewall microservice.
- Microservice handles normalization, tokenization, and request scoring via transformer/GNN models.
- Reinforcement learning policy chooses "block", "challenge", or "allow" actions.
- Dashboard and analytics provide live log monitoring and threat alerts.

## Use Cases

- Detection of zero-day and advanced persistent threats.
- Real-time protection for e-commerce, fintech, and SaaS platforms.
- Research and development of adaptive firewall strategies.

