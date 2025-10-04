# ML System C1

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Objectives](#objectives)
- [Technical Resources](#technical-resources)

## Introduction

**ML System C1** is a production-grade machine learning pipeline that integrates several microservices to process both historical and real-time data with Redpanda (Kafka) and Dockercompose. Data is ingested from raw sources like the Kraken API and processed into OHLC (Open, High, Low, Close) candles through a fully dockerized and production-ready pipeline. This project leverages Hopsworks as a centralized feature store for all data, streamlining the pipeline from data ingestion to real-time prediction, making it well-suited for cutting-edge financial applications.

## Features

- **Real-Time and Historical Data Processing**: Efficiently handles both live and past trade data from Kraken’s API, ensuring a robust input stream for machine learning models.
- **Dockerized Microservices Architecture**: Fully containerized pipeline ensures consistent performance and scalability across different environments.
- **Hopsworks Feature Store Integration**: Uses Hopsworks as a feature store for centralized and efficient data management, simplifying access and organization of features across the pipeline.
- **High Performance with Redpanda Message Broker**: Redpanda facilitates real-time data publishing and consumption across services, ensuring high availability and reliability in data flow.
- **ML Model Deployment**: Prepares the environment to enable machine learning models to make predictions based on real-time data.

## Objectives

The main objective of this project is to build an end-to-end machine learning system that processes real-time data and historical records through a sophisticated feature pipeline.

## Technical Resources

This project includes resources from [Pau Labarta’s course]([https://www.paulabarta.com/](https://www.realworldml.net/)), which provides comprehensive guides and tutorials on building production-ready machine learning systems. Refer to the course materials for more details on best practices in data engineering and model deployment.

