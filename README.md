# Ham Radio Olympics

Olympic-style amateur radio competition with QRZ Logbook integration.

## Features

- **Olympiad/Sport/Match Hierarchy** - Like real Olympics
- **QRZ Integration** - Automatic QSO sync with confirmation status
- **Two Scoring Events per Match**:
  - QSO Race (first three people to log a qualifying QSO)
  - Cool Factor (efficiency: distance/power rewards QRP)
- **Medal System** - Gold (3pts), Silver (2pts), Bronze (1pt), POTA Bonus (+1pt)
- **Work/Activate Modes** - Hunter and activator competitions
- **Separate or Combined Pools** - Configurable per Sport
- **World Records & Personal Bests** - Tracked automatically

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ENCRYPTION_KEY="your-secret-key"
export ADMIN_KEY="your-admin-key"

# Run the server
python -m uvicorn main:app --reload

# Open http://localhost:8000
```

### Run Tests

```bash
pytest tests/ -v
```

## API Endpoints

### Public Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Landing page |
| GET | `/health` | Health check |
| POST | `/register` | Register contestant |
| GET | `/olympiad` | Current Olympiad |
| GET | `/olympiad/sports` | List Sports |
| GET | `/olympiad/sport/{id}` | Sport details |
| GET | `/olympiad/sport/{id}/matches` | List Matches |
| GET | `/olympiad/sport/{id}/match/{id}` | Match leaderboard |
| GET | `/records` | World records |
| GET | `/contestant/{call}` | Contestant profile |
| POST | `/sync` | Trigger QRZ sync |

### Admin Endpoints (require X-Admin-Key header)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/admin` | Dashboard |
| POST | `/admin/olympiad` | Create Olympiad |
| PUT | `/admin/olympiad/{id}` | Update Olympiad |
| DELETE | `/admin/olympiad/{id}` | Delete Olympiad |
| POST | `/admin/olympiad/{id}/activate` | Set active |
| POST | `/admin/olympiad/{id}/sport` | Create Sport |
| PUT | `/admin/sport/{id}` | Update Sport |
| DELETE | `/admin/sport/{id}` | Delete Sport |
| POST | `/admin/sport/{id}/match` | Create Match |
| PUT | `/admin/match/{id}` | Update Match |
| DELETE | `/admin/match/{id}` | Delete Match |
| GET | `/admin/contestants` | List contestants |
| DELETE | `/admin/contestant/{call}` | Remove contestant |

## Example Usage

### Register a Contestant

```bash
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"callsign": "W1ABC", "qrz_api_key": "your-qrz-key"}'
```

### Sync QSOs

```bash
# Sync single contestant
curl -X POST "http://localhost:8000/sync?callsign=W1ABC"

# Sync all contestants
curl -X POST http://localhost:8000/sync
```

### Create an Olympiad (Admin)

```bash
curl -X POST http://localhost:8000/admin/olympiad \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: your-admin-key" \
  -d '{
    "name": "2026 Ham Radio Olympics",
    "start_date": "2026-01-01",
    "end_date": "2026-12-31",
    "qualifying_qsos": 0
  }'
```

### Create a Sport

```bash
curl -X POST http://localhost:8000/admin/olympiad/1/sport \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: your-admin-key" \
  -d '{
    "name": "DX Challenge",
    "target_type": "continent",
    "interval": "monthly",
    "work_enabled": true,
    "activate_enabled": false,
    "separate_pools": false
  }'
```

### Create a Match

```bash
curl -X POST http://localhost:8000/admin/sport/1/match \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: your-admin-key" \
  -d '{
    "start_date": "2026-01-01T00:00:00",
    "end_date": "2026-01-31T23:59:59",
    "target_value": "EU"
  }'
```

## Deployment to Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Create app (first time only)
fly launch --name ham-radio-olympics

# Create persistent volume
fly volumes create ham_olympics_data --size 1 --region iad

# Set secrets
fly secrets set ENCRYPTION_KEY="your-encryption-key"
fly secrets set ADMIN_KEY="your-admin-key"

# Deploy
fly deploy
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| ENCRYPTION_KEY | Key for encrypting QRZ API keys | Yes |
| ADMIN_KEY | Secret for admin authentication | Yes |
| DATABASE_PATH | Path to SQLite database | No (default: ham_olympics.db) |
| PORT | Server port | No (default: 8000) |

## Sport Configuration

| Setting | Options |
|---------|---------|
| target_type | continent, country, park, call, grid |
| interval | daily, weekly, bi-weekly, monthly, quarterly, annually |
| work_enabled | true/false (hunters) |
| activate_enabled | true/false (activators) |
| separate_pools | true/false (per-role medals) |

## Scoring

Here's how points are calculated:

### Medal Points

Each match awards medals for 2 podiums:

  | Medal  | Points |
  |--------|--------|
  | Gold   | 3      |
  | Silver | 2      |
  | Bronze | 1      |

1. QSO Event

  - Winner: First person to make a qualifying contact with the target QSO
  - Medals awarded to the 1st, 2nd, and 3rd earliest confirmed QSOs

2. Cool Factor Event (Power Factor aka QSL)

  - Winner: Highest "cool factor" score
  - Cool Factor formula: distance_km / tx_power_watts
  - Example: 5000 km contact at 5 watts = 1000 cool factor; 1000 km at 1000 watts = 1 cool factor
  - Ties broken by earliest QSO time

### POTA Bonus

  | Scenario                                             | Bonus |
  |------------------------------------------------------|-------|
  | Park-to-Park (target is a park AND you're at a park) | +2    |
  | Target is a park OR you're at a park                 | +1    |
  | No park involvement                                  | +0    |

### Examples

- If you win gold in QSO (3) + silver in QRP (2) + park-to-park bonus (2) = 7 points for that match.
- If you win silver in QSO (2) + bronze in QRP (1) + hunt a park bonus (1) = 4 points for that match.
- If you win bronze in QSO (1) + Gold in QRP (3) + activate a park bonus (1) = 5 points for that match.

### Qualification

The Olympiad can set a minimum number of QSOs required to qualify for medals. If you don't meet the threshold, you appear in standings but aren't eligible for medals.

### Maximum Points per Match
- Single mode or combined pools: 7 pts
- Separate pools: 14 pts (7 per role)

## License

CC BY-NC-SA 4.0
(Use/modify/share, but NOT commercially, and all derivatives must use this same license.)
