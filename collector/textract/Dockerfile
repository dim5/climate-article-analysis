FROM node:alpine

ENV NODE_PORT=8880

RUN apk add --no-cache git

WORKDIR /usr/src/app

COPY package*.json ./
COPY openapi.yaml ./
RUN npm ci


COPY . .
RUN npm run build

EXPOSE $NODE_PORT

ENV NODE_ENV=production
CMD ["node", "dist/server.js"]