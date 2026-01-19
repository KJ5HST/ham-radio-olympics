# Ham Radio Olympics User Guide

**A Competition Platform for Amateur Radio Operators**

Version 1.1 - January 2026

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Competing](#competing)
4. [Scoring & Medals](#scoring--medals)
5. [Teams](#teams)
6. [Records & Personal Bests](#records--personal-bests)
7. [Settings & Preferences](#settings--preferences)
8. [Referee Guide](#referee-guide)
9. [Administrator Guide](#administrator-guide)
10. [Glossary](#glossary)

---

## Introduction

### What is Ham Radio Olympics?

Ham Radio Olympics is a web-based competition platform designed for amateur radio clubs to organize and track radio contests throughout a season. Think of it like the actual Olympics: your club runs an "Olympiad" (a competition season), which contains multiple "Sports" (categories of competition like POTA activations or DX hunting), each with individual "Matches" (specific timed events).

The platform integrates directly with QRZ.com and Logbook of The World (LoTW) to automatically sync your QSO logs, calculate scores, and award medals based on both speed (who made the first qualifying contact) and efficiency (best power-to-distance ratio).

### Key Concepts

**The Olympic Hierarchy**

- **Olympiad:** The overall competition season (e.g., "2026 Ham Radio Olympics"). Spans a defined date range, typically a calendar year.
- **Sport:** A category of competition within the Olympiad (e.g., "POTA Challenge," "DX Hunt," "Grid Square Marathon"). Each sport has its own rules for what counts as a qualifying contact.
- **Match:** A specific timed event within a sport. For example, a monthly POTA match might target K-1234, while the next month targets K-5678.

**User Roles**

- **Competitor:** Any registered amateur radio operator participating in competitions. Can view standings, sync QSOs, join teams, and track personal records.
- **Referee:** A competitor with elevated privileges for specific sports. Can manage matches and view detailed competitor information within their assigned sport(s).
- **Administrator:** Full system access. Can create Olympiads, Sports, and Matches; manage all competitors and teams; and grant referee roles.

### Competition Modes

Each sport can enable one or both competition modes:

- **Work (Hunt):** You're the "hunter" making contacts with the target (e.g., working a POTA station from home).
- **Activate:** You're the "activator" operating from the target location (e.g., activating a park for POTA).

Sports can award medals to hunters and activators in either combined pools (everyone competes together) or separate pools (hunters compete against hunters, activators against activators).

---

## Getting Started

### Registration

To participate in Ham Radio Olympics, you need to register with your callsign and connect at least one logging service (QRZ.com or LoTW).

**Step-by-Step Registration:**

1. Navigate to the Ham Radio Olympics website
2. Click the "Sign Up" button on the landing page
3. Enter your amateur radio callsign (e.g., W1ABC)
4. Create a password for your account
5. Provide your QRZ.com API key and/or LoTW credentials (at least one required)
6. Optionally provide an email address for notifications and password recovery
7. Submit your registration

**Getting Your QRZ.com API Key:**

1. Log in to QRZ.com
2. Navigate to My Logbook > Settings > API tab (requires QRZ XML subscription)
3. Generate or copy your existing API key
4. Your API key is encrypted before storage in Ham Radio Olympics

### Syncing Your QSOs

Ham Radio Olympics automatically pulls your QSO data from QRZ.com and/or LoTW. This includes confirmation status, which is required for medal eligibility.

**Manual Sync:**

1. Navigate to your dashboard
2. Click the "Sync QSOs" button
3. Wait for the sync to complete (this may take a moment depending on log size)

The system will fetch all QSOs from your connected services and match them against active matches in the current Olympiad.

### World Radio League (WRL) Users - IMPORTANT

> **WARNING:** Do NOT use the direct WRL → QRZ integration. There is a bug that strips POTA data (SIG/SIG_INFO fields) during the automatic sync, causing your park hunts and activations to not be credited.

**Correct workflow for WRL users:**

1. **In WRL:** Export your log to an ADIF file (File → Export → ADIF)
2. **In QRZ:** Import that ADIF file manually (My Logbook → Import → Choose File)
3. **In Ham Radio Olympics:** Click "Sync to QRZ" from your Dashboard

This manual export/import process preserves all POTA park references (MY_SIG_INFO for activations, SIG_INFO for hunts) and ensures your contacts are properly credited for competitions.

### Your Dashboard

After logging in, your **Dashboard** is your home base, displaying:

- Total points earned across all sports
- Medal count (Gold, Silver, Bronze)
- Personal best records (current season and all-time)
- Team membership (if any)
- Recent qualifying QSOs
- Quick access to sync, settings, and exports

Your **public profile** is visible to other competitors and shows your achievements, medals, and competition history.

---

## Competing

### Finding Active Competitions

From the main page, you can browse the current Olympiad's sports and matches:

1. Click "Olympiad" to view the current competition season
2. Browse available sports under "Sports"
3. Select a sport to see its matches and rules
4. Click on a specific match to view the leaderboard and target details

### Target Types

Each sport defines what type of target counts as a qualifying contact:

| Target Type | Description |
|-------------|-------------|
| Continent | Contact must be with a station in a specific continent (e.g., EU, AS, AF) |
| Country | Contact must be with a specific DXCC entity |
| Park | Contact must involve a POTA/WWFF park (as hunter or activator) |
| Call | Contact must be with a specific callsign (e.g., special event station) |
| Grid | Contact must be with a station in a specific Maidenhead grid square |
| Any | Any confirmed contact qualifies (useful for general activity periods) |

### Match Events

Every match has two separate medal events:

**QSO Race**

The QSO Race rewards the first three competitors to log a confirmed qualifying contact with the match target. Speed matters! The first confirmed QSO wins gold, second wins silver, third wins bronze.

Your QSO must be confirmed (via QRZ or LoTW) to count for medals.

**Cool Factor (Power Efficiency)**

The Cool Factor event rewards efficient operating. Your score is calculated as:

> **Cool Factor = Distance (km) ÷ TX Power (watts)**

For example:
- 5,000 km contact at 5 watts = 1,000 cool factor
- 1,000 km contact at 1,000 watts = 1 cool factor
- 8,000 km contact at 10 watts = 800 cool factor

The three highest cool factor scores win medals. Ties are broken by earliest QSO time.

### QRP Competitions

Some matches may have a maximum power limit for QRP competitions. Only QSOs at or below the specified power will qualify for that match. Check the match details to see if there's a power restriction.

---

## Scoring & Medals

### Medal Point Values

| Medal | Points |
|-------|--------|
| Gold | 3 points |
| Silver | 2 points |
| Bronze | 1 point |

### POTA Bonuses

Park contacts earn bonus points in addition to any medals:

| Scenario | Bonus Points |
|----------|--------------|
| Park-to-Park (both stations at parks) | +2 points |
| Target is a park OR you're at a park | +1 point |
| No park involvement | +0 points |

### Scoring Examples

- **Example 1:** Gold in QSO Race (3) + Silver in Cool Factor (2) + Park-to-Park (2) = **7 points**
- **Example 2:** Silver in QSO Race (2) + Bronze in Cool Factor (1) + Hunt a park (1) = **4 points**
- **Example 3:** Bronze in QSO Race (1) + Gold in Cool Factor (3) + Activate a park (1) = **5 points**

### Maximum Points Per Match

- **Combined pools:** 7 points maximum (Gold + Gold + P2P bonus)
- **Separate pools:** 14 points maximum (7 points per role)

### Qualification Requirements

The Olympiad administrator can set a minimum QSO threshold for medal eligibility. If you haven't met the minimum, you'll still appear in standings but won't be eligible to receive medals until you reach the threshold.

---

## Teams

### Team Competition

Teams add a collaborative dimension to Ham Radio Olympics. Team standings are calculated by summing the points of all team members across all sports. You can only be a member of one team at a time.

### Creating a Team

Any registered competitor can create a team:

1. Navigate to the Teams page
2. Click "Create Team"
3. Enter a team name and optional description
4. Submit to create the team

When you create a team, you automatically become the **team captain** with special management privileges.

### Joining a Team

There are two ways to join a team:

**Requesting to Join:**

1. Browse existing teams on the Teams page
2. Click on the team you want to join
3. Click "Request to Join"
4. Wait for the team captain to approve your request

If the captain has already sent you an invitation, your join request is automatically accepted.

**Accepting an Invitation:**

Team captains can invite you directly:

1. When invited, you'll see the invitation on your dashboard or the team page
2. Click "Accept" to join the team, or "Reject" to decline
3. If you accept, you immediately become a team member

### Captain Responsibilities

As a team captain, you have additional management capabilities:

| Action | Description |
|--------|-------------|
| Invite Members | Send invitations to specific competitors to join your team |
| Approve Requests | Accept or decline join requests from competitors |
| Remove Members | Remove a member from the team |
| Transfer Captaincy | Hand over the captain role to another team member |
| Update Team Info | Edit the team name and description |
| Delete Team | Permanently delete the team |

### Leaving a Team

To leave your current team:

1. Navigate to the team profile page
2. Click "Leave Team"
3. Confirm your decision

Your individual scores remain intact, but they will no longer contribute to the team total.

> **Note:** Team captains cannot leave the team. You must either transfer captaincy to another member first, or delete the team entirely.

---

## Records & Personal Bests

### World Records

Ham Radio Olympics automatically tracks world records across all competitions. Records are tracked for achievements like:

- Longest distance QSO
- Highest Cool Factor score
- Longest distance QRP contact
- Records by mode (CW, SSB, FT8, etc.)

View current records on the Records page.

### Personal Bests

Your competitor profile tracks your personal best performances:

- **Current Season:** Best performances during the active Olympiad
- **All-Time:** Your best performances across all Olympiads

These are displayed on your dashboard alongside your medal count and total points.

### Medal Standings

The Medal Standings page displays all competitors ranked by their total medal count. The leaderboard shows:

- Gold, Silver, and Bronze medal counts
- Total medals earned
- Total points accumulated

This provides a quick overview of who's leading the competition across all sports.

---

## Settings & Preferences

### Account Settings

Access your settings from the dropdown menu in the navigation bar. Here you can manage:

**Your Name**
- Set your first and last name to display on leaderboards and in the navigation

**Email Address**
- Add or update your email for password recovery and notifications
- Verify your email to enable notifications

**Password**
- Change your account password

**QRZ API Key**
- Add, update, or remove your QRZ Logbook API key

**LoTW Credentials**
- Add, update, or remove your Logbook of The World credentials

### Display Preferences

Customize how information is displayed throughout the site:

**Distance Unit**
- Choose between Kilometers (km) or Miles (mi)
- All distances on the site will be converted to your preferred unit

**Time Display**
- **UTC:** All times shown in Coordinated Universal Time
- **Local Time:** Times are converted to local time based on the QSO's grid square location

### Email Notifications

If you've added and verified an email address, you can receive notifications for:

- **Medal notifications:** When you earn a new medal
- **Weekly match digest:** Upcoming matches in sports you've entered
- **Record notifications:** When you set a new world record

You can enable or disable each notification type individually, or turn off all notifications at once.

### Dark Mode

Use the moon/sun icon in the navigation bar to toggle between light and dark mode. The site also respects your system's color scheme preference.

---

## Referee Guide

### Referee Role Overview

Referees are competitors with elevated privileges for specific sports. The referee role is granted by an administrator and allows you to manage matches and view detailed competitor information within your assigned sport(s).

### Accessing the Referee Dashboard

Once granted referee status, you can access the referee dashboard from the dropdown menu. This provides an overview of your assigned sports and any pending tasks.

### Managing Matches

As a referee for a sport, you can:

- View all matches in your assigned sport(s)
- Create new matches within your assigned sport(s)
- Edit existing matches (dates, targets, power limits)
- Delete matches if needed
- Access detailed competitor information and QSO logs
- Monitor match progress and standings
- Disqualify competitors from your sport(s)

---

## Administrator Guide

### Admin Dashboard

Administrator access provides full control over the competition platform. Access the admin dashboard from the dropdown menu after logging in with an admin account.

### Creating an Olympiad

An Olympiad is the top-level container for a competition season. To create one:

1. Access the admin dashboard
2. Click "Create Olympiad"
3. Provide name, start date, end date, and qualifying QSO minimum
4. After creation, activate the Olympiad to make it current

Only one Olympiad can be active at a time.

### Creating Sports

Sports define categories of competition. Configuration options include:

| Setting | Options / Description |
|---------|----------------------|
| Target Type | continent, country, park, call, grid, any |
| Interval | daily, weekly, bi-weekly, monthly, quarterly, annually |
| Work Enabled | Allows hunting (working stations) |
| Activate Enabled | Allows activating (being the target) |
| Separate Pools | If enabled, hunters and activators compete separately |

### Creating Matches

Matches are timed events within a sport. Each match requires:

- **Start Date/Time:** When the match begins (UTC)
- **End Date/Time:** When the match ends
- **Target Value:** The specific target (e.g., "EU" for continent, "K-1234" for park)

Optional match settings:

- **Allowed Modes:** Restrict the match to specific modes (e.g., CW only)
- **Max Power (watts):** Limit the match to QRP contacts only

### Managing Competitors

Administrators can:

- View all registered competitors
- Enable or disable competitor accounts
- Grant or revoke admin privileges
- Grant or revoke referee roles
- Assign referees to specific sports
- Reset competitor passwords
- Remove competitors if necessary

### Managing Teams

Administrators have full team management capabilities:

- View all teams
- Create teams directly
- Add or remove team members
- Transfer team captaincy
- Delete teams if necessary

### Site Settings

Administrators can customize the site appearance:

- **Theme:** Choose from multiple visual themes
- **Site Name:** Customize the site title
- **Tagline:** Set a custom tagline displayed in the header
- **QRZ Credentials:** Configure QRZ XML API for callsign lookups

---

## Glossary

| Term | Definition |
|------|------------|
| **Cool Factor** | A scoring metric calculated as distance divided by transmit power, rewarding efficient operating. |
| **DXCC** | DX Century Club, an ARRL award program. Also refers to the list of recognized "entities" (countries/territories) for award purposes. |
| **Grid Square** | A Maidenhead Locator System designation for geographic location (e.g., FN31). |
| **LoTW** | Logbook of The World, an ARRL-operated QSO confirmation service. |
| **Match** | A timed competition event within a sport, targeting a specific entity. |
| **Olympiad** | A competition season containing multiple sports. |
| **P2P (Park-to-Park)** | A contact where both stations are activating parks, earning bonus points. |
| **POTA** | Parks on the Air, an amateur radio program encouraging portable operations from parks. |
| **QRP** | Low-power amateur radio operation, typically 5 watts or less for CW and 10 watts or less for SSB. |
| **QRZ** | QRZ.com, a callsign lookup database and online logbook service. |
| **QSO** | A radio contact between two stations. |
| **QSO Race** | A medal event rewarding the first three confirmed contacts with a match target. |
| **Sport** | A category of competition within an Olympiad (e.g., DX Challenge, POTA). |

---

*End of User Guide*
