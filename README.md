# Ham Radio Olympics

Amateur radio competition server with LoTW and QRZ Logbook integration.

**[Full Documentation](docs/README.md)** | **[Changelog](docs/CHANGELOG.md)** | **[Email Setup](docs/EMAIL_SETUP.md)**

## Quick Start

```bash
pip install -r requirements.txt
export ENCRYPTION_KEY="your-secret-key"
export ADMIN_KEY="your-admin-key"
python -m uvicorn main:app --reload
```

See [docs/README.md](docs/README.md) for complete documentation.
