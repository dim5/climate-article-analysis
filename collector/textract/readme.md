# Textract

This is a service to extract articles from news sites.

## How to start

First, install the dependencies

```bash
npm install
```

Then you can either use Docker or NPM to start the service.

### NPM

```bash
npm start
```

### Docker

```bash
docker build -t textract:dev .
docker run -p 8880:8880 textract:dev
```

## Docs

After starting the project, visit `/swagger`.
