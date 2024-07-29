# Docker Cheatsheet

### Builing and Running

```bash
# Build
make build

# Run
make run
```

### Show Disk Space Used

```bash
docker system df
```

# Clear Disk Spage (and also network interfaces created by Docker)

```bash
docker image prune -f
docker container prune -f
docker buildx prune -f
docker network prune -f

# Or, remove everything in one go:
docker system prune -f
```
