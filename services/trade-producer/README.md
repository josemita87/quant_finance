# Trade Producer Service

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)

- [Contributing](#contributing)

## Overview

The `trade-producer` service generates trade data and publishes it to a message broker for consumption by other services in the `ml_system_c1` project. This service is dockerized and designed for efficiency. It is also capable of handling high volumes of trade data in real time, making it a critical component of a sophisticated machine learning system.

## Current Capabilities

- **Real-Time Trade Data Generation**: Generates and publishes trade data instantaneously, leveraging Kraken Batch API.
- **Robust Message Publishing**: Facilitates seamless data flow by connecting to a message broker (redpanda).
- **Configurable Parameters**: Allows customization of trade data generation settings to accommodate various use cases.
- **High Performance and Scalability**: Engineered to manage increasing volumes of trade data without degradation in performance, thanks to dockerization and usage of redpanda services..

## Goal

The objective is to develop an end-to-end machine learning system that leverages sophisticated data pipelines to ingest, transform, and load data for model training. This project serves as a foundational element in creating a robust, data-driven environment that supports innovative financial solutions.

sdf
## Additional Resources

For further documentation and examples, please refer to the course materials provided by Pau Labarta. Comprehensive guides, tutorials, and additional resources can be found on his website.