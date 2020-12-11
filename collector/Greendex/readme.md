# Greendex

This is a server application for managing the article collection process.

## How to start

You can either use Visual Studio, the dotnet CLI, or Docker to start this project.

Of course, you'll need to supply your own connection string and other secrets, see `Greendex\appsettings(.Development).json` for more.

### Docker

```bash
docker build -t greendex:dev .
docker run -p 8888:80 greendex:dev
```
