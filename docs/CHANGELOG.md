# Changelog

All notable changes to Ham Radio Olympics are documented in this file.

## [1.2.1] - 2026-02-20

### Added
- **Auto-Deploy from GitHub**: GitHub Actions workflow (`.github/workflows/deploy.yml`) automatically deploys to Fly.io on push to `main`
- **fly.toml committed to repo**: Removed from `.gitignore` so GitHub Actions can deploy (contains no secrets)

### Fixed
- **Push Notification Recovery**: Settings page now auto-re-registers push subscriptions when the server has lost the record but the browser still has one
- **Email Notifications After Sync**: Medal notification emails now sent automatically after every sync (previously only triggered manually from admin)
- **Batched Discord Notifications**: Medal award announcements are now consolidated into a single Discord message per sync instead of individual messages
- **Orphaned Records**: Records are now recomputed after QSO deletions to prevent stale `qso_id` references
- **Triathlon Standings**: Competitors who withdraw from all sports are now correctly excluded from triathlon standings

## [1.2.0] - 2026-01-11

### Added
- **Referee Dashboard**: Referees now have a dedicated navigation link and dashboard to manage their assigned sports
- **Medal Links**: Clicking medal counts on dashboard scrolls to medals section; each medal row links to its match
- **Team Sport Standings**: Team profile pages now show aggregated standings across all sports
- **Auto-Recompute Team Standings**: Team standings automatically recalculate when members are added/removed

### Fixed
- **Medal Count Bug**: Fixed cartesian product issue that showed inflated medal counts when multiple QSOs had same timestamp
- **Team Display**: Members can now see their team on the "My Team" section of their dashboard
- **Admin Teams Member Count**: Fixed duplicate counting in admin teams list
- **Team Creation Errors**: Improved error handling with toast notifications instead of raw JSON

### Changed
- **Team Profile Medals**: Changed from "1G 2S" format to emoji icons (🥇1 🥈2)
- **Qualifying QSOs Display**: Clarified "0" means no minimum requirement

## [1.1.0] - 2026-01-10

### Added
- **Match Status Badges**: Sport pages now show Active/Ended/Upcoming badges for each match
- **Sync Progress Indicators**: Success toast shows sync results (e.g., "Synced 50 QSOs (15 new, 35 updated)")
- **Tie-Breaker Test Coverage**: 8 new tests covering medal tie-breaking scenarios
- **Match Date Validation**: Matches must now fall within their Olympiad's date range
- **Duplicate QSO Prevention**: Database constraint prevents duplicate QSOs from being inserted
- **API Rate Limits**: All `/api/v1/*` endpoints now have rate limiting (60/minute)
- **OpenAPI Documentation**: Interactive API docs available at `/docs` and `/redoc`
- **Configuration Template**: `.env.example` file documents all environment variables
- **Medal-QSO Linking**: Dashboard shows DX callsign next to each medal with tooltip details

### Changed
- **Medal Display**: Changed from text boxes to emoji medals (🥇🥈🥉)
- **Loading States**: Buttons show loading spinner during async operations
- **Admin Authentication**: Admin key now only accepted via `X-Admin-Key` header (not query params)
- **Record Updates**: Now use exclusive database transactions to prevent race conditions

### Security
- Removed admin_key query parameter support (security: keys were logged in URLs)
- Added unique constraint on QSOs to prevent data corruption

### Performance
- Added database index on `records(callsign)` for faster personal best lookups

## [1.0.0] - 2026-01-01

### Added
- Initial release of Ham Radio Olympics
- Olympiad/Sport/Match hierarchy for organizing competitions
- QSO sync from QRZ Logbook and LoTW
- Medal system with Gold/Silver/Bronze for QSO Race and Cool Factor
- POTA bonus scoring
- Team competitions with multiple scoring methods
- Personal bests and world records tracking
- Admin dashboard for managing competitions
- Referee role with sport-specific permissions
- Dark mode support
- Mobile-responsive design

### Technical
- FastAPI backend with SQLite database
- Jinja2 templates for server-side rendering
- Session-based authentication with CSRF protection
- Rate limiting on sensitive endpoints
- Encrypted storage for API credentials
