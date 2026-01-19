# Ham Radio Olympics User Guide

**A Competition Platform for Amateur Radio Operators**

*Version 1.0 — January 2026*

---

## Table of Contents

1. [Introduction](#1-introduction)
   - [What is Ham Radio Olympics?](#11-what-is-ham-radio-olympics)
   - [Key Concepts](#12-key-concepts)
   - [Competition Modes](#13-competition-modes)
2. [Getting Started (Competitors)](#2-getting-started-competitors)
   - [Registration](#21-registration)
   - [Syncing Your QSOs](#22-syncing-your-qsos)
   - [Your Competitor Profile](#23-your-competitor-profile)
3. [Competing](#3-competing)
   - [Finding Active Competitions](#31-finding-active-competitions)
   - [Target Types](#32-target-types)
   - [Match Events](#33-match-events)
4. [Scoring & Medals](#4-scoring--medals)
   - [Medal Point Values](#41-medal-point-values)
   - [POTA Bonuses](#42-pota-bonuses)
   - [Scoring Examples](#43-scoring-examples)
   - [Maximum Points Per Match](#44-maximum-points-per-match)
   - [Qualification Requirements](#45-qualification-requirements)
5. [Teams](#5-teams)
   - [Team Competition](#51-team-competition)
   - [Creating a Team](#52-creating-a-team)
   - [Joining a Team](#53-joining-a-team)
   - [Captain Responsibilities](#54-captain-responsibilities)
   - [Leaving a Team](#55-leaving-a-team)
6. [Records & Personal Bests](#6-records--personal-bests)
   - [World Records](#61-world-records)
   - [Personal Bests](#62-personal-bests)
   - [Medal Standings](#63-medal-standings)
7. [Referee Guide](#7-referee-guide)
   - [Referee Role Overview](#71-referee-role-overview)
   - [Accessing the Referee Dashboard](#72-accessing-the-referee-dashboard)
   - [Managing Matches](#73-managing-matches)
8. [Administrator Guide](#8-administrator-guide)
   - [Admin Authentication](#81-admin-authentication)
   - [Creating an Olympiad](#82-creating-an-olympiad)
   - [Creating Sports](#83-creating-sports)
   - [Creating Matches](#84-creating-matches)
   - [Managing Competitors](#85-managing-competitors)
   - [Managing Teams](#86-managing-teams)
9. [Deployment & Configuration](#9-deployment--configuration)
   - [Environment Variables](#91-environment-variables)
   - [Local Development](#92-local-development)
   - [Fly.io Deployment](#93-flyio-deployment)
   - [Running Tests](#94-running-tests)
- [Appendix A: API Reference](#appendix-a-api-reference)
- [Appendix B: Glossary](#appendix-b-glossary)

---

## 1. Introduction

### 1.1 What is Ham Radio Olympics?

Ham Radio Olympics is a web-based competition platform designed for amateur radio clubs to organize and track radio contests throughout a season. Think of it like the actual Olympics: your club runs an "Olympiad" (a competition season), which contains multiple "Sports" (categories of competition like POTA activations or DX hunting), each with individual "Matches" (specific timed events).

The platform integrates directly with QRZ.com and Logbook of The World (LoTW) to automatically sync your QSO logs, calculate scores, and award medals based on both speed (who made the first qualifying contact) and efficiency (best power-to-distance ratio).

### 1.2 Key Concepts

#### The Olympic Hierarchy

- **Olympiad:** The overall competition season (e.g., "2026 Ham Radio Olympics"). Spans a defined date range, typically a calendar year.
- **Sport:** A category of competition within the Olympiad (e.g., "POTA Challenge," "DX Hunt," "Grid Square Marathon"). Each sport has its own rules for what counts as a qualifying contact.
- **Match:** A specific timed event within a sport. For example, a monthly POTA match might target K-1234, while the next month targets K-5678.

#### User Roles

- **Competitor:** Any registered amateur radio operator participating in competitions. Can view standings, sync QSOs, join teams, and track personal records.
- **Referee:** A competitor with elevated privileges for specific sports. Can manage matches and view detailed competitor information within their assigned sport(s).
- **Administrator:** Full system access. Can create Olympiads, Sports, and Matches; manage all competitors and teams; and grant referee roles.

### 1.3 Competition Modes

Each sport can enable one or both competition modes:

- **Work (Hunt):** You're the "hunter" making contacts with the target (e.g., working a POTA station from home).
- **Activate:** You're the "activator" operating from the target location (e.g., activating a park for POTA).

Sports can award medals to hunters and activators in either combined pools (everyone competes together) or separate pools (hunters compete against hunters, activators against activators).

---

## 2. Getting Started (Competitors)

### 2.1 Registration

To participate in Ham Radio Olympics, you need to register with your callsign and connect at least one logging service (QRZ.com or LoTW).

#### Step-by-Step Registration

1. Navigate to the Ham Radio Olympics website
2. Click the "Sign Up" button on the landing page
3. Enter your amateur radio callsign (e.g., W1ABC)
4. Create a password for your account
5. Provide your QRZ.com API key and/or LoTW credentials (at least one required)
6. Optionally provide an email address for password recovery
7. Submit your registration

#### Getting Your QRZ.com API Key

1. Log in to QRZ.com
2. Navigate to My Logbook > Settings > API tab (requires QRZ XML subscription)
3. Generate or copy your existing API key
4. Your API key is encrypted before storage in Ham Radio Olympics

### 2.2 Syncing Your QSOs

Ham Radio Olympics automatically pulls your QSO data from QRZ.com and/or LoTW. This includes confirmation status, which is required for medal eligibility.

#### Manual Sync

To manually trigger a sync of your QSOs:

1. Navigate to your dashboard or the Sync page (`/sync`)
2. Click the "Sync QSOs" button
3. Wait for the sync to complete (this may take a moment depending on log size)

The system will fetch all QSOs from your connected services and match them against active matches in the current Olympiad.

#### World Radio League (WRL) Users - IMPORTANT

> **WARNING:** Do NOT use the direct WRL → QRZ integration. There is a bug that strips POTA data (SIG/SIG_INFO fields) during the automatic sync, causing your park hunts and activations to not be credited.

**Correct workflow for WRL users:**

1. **In WRL:** Export your log to an ADIF file (File → Export → ADIF)
2. **In QRZ:** Import that ADIF file manually (My Logbook → Import → Choose File)
3. **In Ham Radio Olympics:** Click "Sync to QRZ" from your Dashboard

This manual export/import process preserves all POTA park references (MY_SIG_INFO for activations, SIG_INFO for hunts) and ensures your contacts are properly credited for competitions.

### 2.3 Your Dashboard and Profile

After logging in, your **Dashboard** (`/dashboard`) is your home base, displaying:

- Total points earned across all sports
- Medal count (Gold, Silver, Bronze)
- Personal best records
- Team membership (if any)
- Recent qualifying QSOs
- Quick access to sync, settings, and exports

Your **public profile** is visible to other competitors at `/competitor/YOUR_CALLSIGN` (e.g., `/competitor/W1ABC`).

---

## 3. Competing

### 3.1 Finding Active Competitions

From the main page, you can browse the current Olympiad's sports and matches:

1. Click "Olympiad" to view the current competition season
2. Browse available sports under "Sports"
3. Select a sport to see its matches and rules
4. Click on a specific match to view the leaderboard and target details

### 3.2 Target Types

Each sport defines what type of target counts as a qualifying contact:

| Target Type | Description |
|-------------|-------------|
| Continent | Contact must be with a station in a specific continent (e.g., EU, AS, AF) |
| Country | Contact must be with a specific DXCC entity |
| Park | Contact must involve a POTA/WWFF park (as hunter or activator) |
| Call | Contact must be with a specific callsign (e.g., special event station) |
| Grid | Contact must be with a station in a specific Maidenhead grid square |
| Any | Any confirmed contact qualifies (useful for general activity periods) |

### 3.3 Match Events

Every match has two separate medal events:

#### QSO Race

The QSO Race rewards the first three competitors to log a confirmed qualifying contact with the match target. Speed matters! The first confirmed QSO wins gold, second wins silver, third wins bronze.

Your QSO must be confirmed (via QRZ or LoTW) to count for medals.

#### Cool Factor (Power Efficiency)

The Cool Factor event rewards efficient operating. Your score is calculated as:

> **Cool Factor = Distance (km) ÷ TX Power (watts)**

For example:

- 5,000 km contact at 5 watts = 1,000 cool factor
- 1,000 km contact at 1,000 watts = 1 cool factor
- 8,000 km contact at 10 watts = 800 cool factor

The three highest cool factor scores win medals. Ties are broken by earliest QSO time.

---

## 4. Scoring & Medals

### 4.1 Medal Point Values

| Medal | Points |
|-------|--------|
| Gold | 3 points |
| Silver | 2 points |
| Bronze | 1 point |

### 4.2 POTA Bonuses

Park contacts earn bonus points in addition to any medals:

| Scenario | Bonus Points |
|----------|--------------|
| Park-to-Park (both stations at parks) | +2 points |
| Target is a park OR you're at a park | +1 point |
| No park involvement | +0 points |

### 4.3 Scoring Examples

- **Example 1:** Gold in QSO Race (3) + Silver in Cool Factor (2) + Park-to-Park (2) = **7 points**
- **Example 2:** Silver in QSO Race (2) + Bronze in Cool Factor (1) + Hunt a park (1) = **4 points**
- **Example 3:** Bronze in QSO Race (1) + Gold in Cool Factor (3) + Activate a park (1) = **5 points**

### 4.4 Maximum Points Per Match

- **Combined pools:** 7 points maximum (Gold + Gold + P2P bonus)
- **Separate pools:** 14 points maximum (7 points per role)

### 4.5 Qualification Requirements

The Olympiad administrator can set a minimum QSO threshold for medal eligibility. If you haven't met the minimum, you'll still appear in standings but won't be eligible to receive medals until you reach the threshold.

---

## 5. Teams

### 5.1 Team Competition

Teams add a collaborative dimension to Ham Radio Olympics. Team standings are calculated by summing the points of all team members across all sports. You can only be a member of one team at a time.

### 5.2 Creating a Team

Any registered competitor can create a team:

1. Navigate to the Teams page (`/teams`)
2. Click "Create Team"
3. Enter a team name and optional description
4. Submit to create the team

When you create a team, you automatically become the **team captain** with special management privileges.

### 5.3 Joining a Team

There are two ways to join a team:

#### Requesting to Join

1. Browse existing teams at `/teams`
2. Click on the team you want to join
3. Click "Request to Join"
4. Wait for the team captain to approve your request

If the captain has already sent you an invitation, your join request is automatically accepted.

#### Accepting an Invitation

Team captains can invite you directly:

1. When invited, you'll see the invitation on your dashboard or the team page
2. Click "Accept" to join the team, or "Reject" to decline
3. If you accept, you immediately become a team member

If you've already sent a join request to that team, accepting the invitation is automatic.

### 5.4 Captain Responsibilities

As a team captain, you have additional management capabilities:

| Action | Description |
|--------|-------------|
| **Invite Members** | Send invitations to specific competitors to join your team |
| **Approve Requests** | Accept or decline join requests from competitors |
| **Remove Members** | Remove a member from the team |
| **Transfer Captaincy** | Hand over the captain role to another team member |
| **Update Team Info** | Edit the team name and description |
| **Delete Team** | Permanently delete the team |

To manage your team, navigate to your team's profile page at `/team/{id}`.

### 5.5 Leaving a Team

To leave your current team:

1. Navigate to the team profile page
2. Click "Leave Team"
3. Confirm your decision

Your individual scores remain intact, but they will no longer contribute to the team total.

**Note:** Team captains cannot leave the team. You must either transfer captaincy to another member first, or delete the team entirely.

---

## 6. Records & Personal Bests

### 6.1 World Records

Ham Radio Olympics automatically tracks world records across all competitions. Records are tracked for achievements like:

- Highest Cool Factor score in a single QSO
- Fastest qualifying QSO after match start
- Longest distance QRP contact

View current records at `/records`.

### 6.2 Personal Bests

Your competitor profile tracks your personal best performances, allowing you to measure your improvement over time. These are displayed on your profile page alongside your medal count and total points.

### 6.3 Medal Standings

The Medal Standings page (`/medals`) displays all competitors ranked by their total medal count. The leaderboard shows:

- Gold, Silver, and Bronze medal counts
- Total medals earned
- Total points accumulated

This provides a quick overview of who's leading the competition across all sports.

---

## 7. Referee Guide

### 7.1 Referee Role Overview

Referees are competitors with elevated privileges for specific sports. The referee role is granted by an administrator and allows you to manage matches and view detailed competitor information within your assigned sport(s).

### 7.2 Accessing the Referee Dashboard

Once granted referee status, you can access the referee dashboard at `/referee`. This provides an overview of your assigned sports and any pending tasks.

### 7.3 Managing Matches

As a referee for a sport, you can:

- View all matches in your assigned sport(s)
- Create new matches within your assigned sport(s)
- Edit existing matches (dates, targets, power limits)
- Delete matches if needed
- Access detailed competitor information and QSO logs
- Monitor match progress and standings
- Disqualify competitors from your sport(s)

---

## 8. Administrator Guide

### 8.1 Admin Authentication

Administrator access can be granted in two ways:

1. **Admin User Account:** A competitor account with admin privileges. Admins can grant admin status to other users via the admin dashboard.
2. **API Key Header:** For programmatic access, include the `X-Admin-Key` header with requests. This key is configured as an environment variable (`ADMIN_KEY`) during deployment.

To access the admin dashboard, log in with an admin account and navigate to `/admin`.

### 8.2 Creating an Olympiad

An Olympiad is the top-level container for a competition season. To create one:

1. Access the admin dashboard at `/admin`
2. Click "Create Olympiad"
3. Provide name, start date, end date, and qualifying QSO minimum
4. After creation, activate the Olympiad to make it current

Only one Olympiad can be active at a time.

### 8.3 Creating Sports

Sports define categories of competition. Configuration options include:

| Setting | Options / Description |
|---------|----------------------|
| Target Type | continent, country, park, call, grid, any |
| Interval | daily, weekly, bi-weekly, monthly, quarterly, annually |
| Work Enabled | true/false — allows hunting (working stations) |
| Activate Enabled | true/false — allows activating (being the target) |
| Separate Pools | true/false — if true, hunters and activators compete separately |

### 8.4 Creating Matches

Matches are timed events within a sport. Each match requires:

- **Start Date/Time:** When the match begins (UTC recommended)
- **End Date/Time:** When the match ends
- **Target Value:** The specific target (e.g., "EU" for continent, "K-1234" for park)

Optional match settings:

- **Allowed Modes:** Override sport-level mode restrictions for this match only
- **Max Power (watts):** Limit the match to QRP contacts — only QSOs at or below this power will qualify

### 8.5 Managing Competitors

Administrators can:

- View all registered competitors at `/admin/contestants`
- Remove competitors if necessary
- Grant referee roles for specific sports
- Revoke referee roles when needed

### 8.6 Managing Teams

Administrators have full team management capabilities:

- View all teams at `/admin/teams`
- Create teams directly
- Delete teams if necessary

---

## 9. Deployment & Configuration

### 9.1 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ENCRYPTION_KEY` | Secret key for encrypting QRZ API keys at rest | Yes |
| `ADMIN_KEY` | Secret for administrator authentication | Yes |
| `DATABASE_PATH` | Path to SQLite database file | No (default: `ham_olympics.db`) |
| `PORT` | Server port | No (default: `8000`) |

### 9.2 Local Development

To run Ham Radio Olympics locally for development or testing:

1. Clone the repository from GitHub
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables (`ENCRYPTION_KEY`, `ADMIN_KEY`)
4. Run: `python -m uvicorn main:app --reload`
5. Open http://localhost:8000 in your browser

### 9.3 Fly.io Deployment

Ham Radio Olympics includes a Dockerfile and `fly.toml` for easy deployment to Fly.io:

1. Install flyctl: `curl -L https://fly.io/install.sh | sh`
2. Login: `fly auth login`
3. Create app: `fly launch --name ham-radio-olympics`
4. Create persistent volume: `fly volumes create ham_olympics_data --size 1 --region iad`
5. Set secrets: `fly secrets set ENCRYPTION_KEY="..." ADMIN_KEY="..."`
6. Deploy: `fly deploy`

### 9.4 Running Tests

The project includes a comprehensive test suite. Run tests with:

```bash
pytest tests/ -v
```

---

## Appendix A: API Reference

### A.1 Public Endpoints (No Authentication Required)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Landing page |
| GET | `/health` | Health check |
| GET | `/signup` | Registration page |
| POST | `/signup` | Register new competitor |
| GET | `/login` | Login page |
| POST | `/login` | Authenticate user |
| GET | `/forgot-password` | Password reset request page |
| POST | `/forgot-password` | Request password reset email |
| GET | `/reset-password/{token}` | Password reset page |
| POST | `/reset-password/{token}` | Reset password with token |
| GET | `/verify-email/{token}` | Verify email address |

### A.2 Authenticated User Endpoints

These endpoints require a logged-in user:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/logout` | Log out current user |
| GET | `/dashboard` | User dashboard |
| GET | `/settings` | User settings page |
| POST | `/settings/name` | Update display name |
| POST | `/settings/email` | Update email address |
| POST | `/settings/password` | Change password |
| POST | `/settings/qrz-key` | Set QRZ API key |
| DELETE | `/settings/qrz-key` | Remove QRZ API key |
| POST | `/settings/lotw` | Set LoTW credentials |
| DELETE | `/settings/lotw` | Remove LoTW credentials |
| GET | `/olympiad` | Current Olympiad details |
| GET | `/olympiad/sports` | List Sports |
| GET | `/olympiad/sport/{id}` | Sport details |
| GET | `/olympiad/sport/{id}/participants` | Sport participants |
| GET | `/olympiad/sport/{id}/matches` | List Matches |
| GET | `/olympiad/sport/{id}/match/{id}` | Match leaderboard |
| POST | `/sport/{id}/enter` | Enter a sport |
| POST | `/sport/{id}/leave` | Leave a sport |
| GET | `/medals` | Medal standings |
| GET | `/records` | World records |
| GET | `/competitor/{call}` | Competitor profile |
| GET | `/sync` | Sync page |
| POST | `/sync` | Trigger QRZ/LoTW sync |
| GET | `/export/qsos` | Export QSOs (CSV) |
| GET | `/export/medals` | Export medals (CSV) |
| GET | `/teams` | Team listings |
| GET | `/team/{id}` | Team profile |
| POST | `/team` | Create team |
| PUT | `/team/{id}` | Update team (captain only) |
| DELETE | `/team/{id}` | Delete team (captain only) |
| POST | `/team/{id}/request` | Request to join team |
| POST | `/team/{id}/invite/{call}` | Invite member (captain only) |
| POST | `/team/{id}/approve/{call}` | Approve join request (captain only) |
| POST | `/team/{id}/decline/{call}` | Decline join request (captain only) |
| POST | `/team/{id}/accept` | Accept team invitation |
| POST | `/team/{id}/reject` | Reject team invitation |
| POST | `/team/{id}/leave` | Leave team |
| POST | `/team/{id}/remove/{call}` | Remove member (captain only) |
| POST | `/team/{id}/transfer/{call}` | Transfer captain role |

### A.3 Referee Endpoints

These endpoints require referee role for the associated sport:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/referee` | Referee dashboard |
| GET | `/admin/sport/{id}/matches` | Manage sport matches |
| GET | `/admin/sport/{id}/competitors` | View sport competitors |
| POST | `/admin/sport/{id}/competitor/{call}/disqualify` | Disqualify competitor from sport |

### A.4 Admin Endpoints

These endpoints require administrator privileges (admin user login or `X-Admin-Key` header):

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/admin` | Admin dashboard |
| GET | `/admin/audit-log` | View audit log |
| GET | `/admin/backup` | Download database backup |
| POST | `/admin/recompute-records` | Recompute all medals and records |
| POST | `/admin/olympiad` | Create Olympiad |
| GET | `/admin/olympiad/{id}` | Get Olympiad details |
| PUT | `/admin/olympiad/{id}` | Update Olympiad |
| DELETE | `/admin/olympiad/{id}` | Delete Olympiad |
| POST | `/admin/olympiad/{id}/activate` | Set as active Olympiad |
| POST | `/admin/olympiad/{id}/deactivate` | Deactivate Olympiad |
| GET | `/admin/olympiad/{id}/sports` | List sports in Olympiad |
| POST | `/admin/olympiad/{id}/sport` | Create Sport |
| GET | `/admin/sport/{id}` | Get Sport details |
| PUT | `/admin/sport/{id}` | Update Sport |
| DELETE | `/admin/sport/{id}` | Delete Sport |
| POST | `/admin/sport/{id}/match` | Create Match |
| GET | `/admin/match/{id}` | Get Match details |
| PUT | `/admin/match/{id}` | Update Match |
| DELETE | `/admin/match/{id}` | Delete Match |
| GET | `/admin/competitors` | List competitors |
| GET | `/admin/export/competitors` | Export competitors (CSV) |
| GET | `/admin/export/standings/{id}` | Export standings (CSV) |
| POST | `/admin/competitors/bulk-disable` | Bulk disable competitors |
| POST | `/admin/competitors/bulk-enable` | Bulk enable competitors |
| POST | `/admin/competitors/bulk-delete` | Bulk delete competitors |
| POST | `/admin/competitor/{call}/disable` | Disable competitor |
| POST | `/admin/competitor/{call}/enable` | Enable competitor |
| POST | `/admin/competitor/{call}/set-admin` | Grant admin role |
| POST | `/admin/competitor/{call}/remove-admin` | Revoke admin role |
| POST | `/admin/competitor/{call}/set-referee` | Grant referee role |
| POST | `/admin/competitor/{call}/remove-referee` | Revoke referee role |
| POST | `/admin/competitor/{call}/assign-sport/{id}` | Assign referee to sport |
| DELETE | `/admin/competitor/{call}/assign-sport/{id}` | Remove referee from sport |
| POST | `/admin/competitor/{call}/disqualify` | Disqualify competitor globally |
| POST | `/admin/competitor/{call}/reset-password` | Reset competitor password |
| DELETE | `/admin/competitor/{call}` | Delete competitor |
| GET | `/admin/teams` | List all teams |
| POST | `/admin/team` | Create team |
| DELETE | `/admin/team/{id}` | Delete team |
| POST | `/admin/team/{id}/add/{call}` | Add member to team |
| POST | `/admin/team/{id}/remove/{call}` | Remove member from team |
| POST | `/admin/team/{id}/transfer/{call}` | Transfer team captain |

---

## Appendix B: Glossary

- **Cool Factor:** A scoring metric calculated as distance divided by transmit power, rewarding efficient operating.
- **DXCC:** DX Century Club, an ARRL award program. Also refers to the list of recognized "entities" (countries/territories) for award purposes.
- **Grid Square:** A Maidenhead Locator System designation for geographic location (e.g., FN31).
- **LoTW:** Logbook of The World, an ARRL-operated QSO confirmation service.
- **Match:** A timed competition event within a sport, targeting a specific entity.
- **Olympiad:** A competition season containing multiple sports.
- **P2P (Park-to-Park):** A contact where both stations are activating parks, earning bonus points.
- **POTA:** Parks on the Air, an amateur radio program encouraging portable operations from parks.
- **QRP:** Low-power amateur radio operation, typically 5 watts or less for CW and 10 watts or less for SSB. Matches can set a max power limit to create QRP-only competitions.
- **QRZ:** QRZ.com, a callsign lookup database and online logbook service.
- **QSO:** A radio contact between two stations.
- **QSO Race:** A medal event rewarding the first three confirmed contacts with a match target.
- **Sport:** A category of competition within an Olympiad (e.g., DX Challenge, POTA).

---

*— End of Document —*
