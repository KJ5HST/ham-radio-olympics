# Ham Radio Olympics User Guide

**A Competition Platform for Amateur Radio Operators**

*Version 1.7 — February 2026*

---

## Table of Contents

1. [Introduction](#1-introduction)
   - [What is Ham Radio Olympics?](#11-what-is-ham-radio-olympics)
   - [Key Concepts](#12-key-concepts)
   - [Competition Modes](#13-competition-modes)
2. [Getting Started (Competitors)](#2-getting-started-competitors)
   - [Registration](#21-registration)
   - [Syncing Your QSOs](#22-syncing-your-qsos)
   - [Your Dashboard and Profile](#23-your-dashboard-and-profile)
3. [Competing](#3-competing)
   - [Finding Active Competitions](#31-finding-active-competitions)
   - [Target Types](#32-target-types)
   - [Match Events](#33-match-events)
   - [QRP Competitions](#34-qrp-competitions)
   - [QSO Disqualifications](#35-qso-disqualifications)
4. [Scoring & Medals](#4-scoring--medals)
   - [Medal Point Values](#41-medal-point-values)
   - [One Podium Spot Per Competitor](#42-one-podium-spot-per-competitor)
   - [POTA Bonuses](#43-pota-bonuses)
   - [Scoring Examples](#44-scoring-examples)
   - [Maximum Points Per Match](#45-maximum-points-per-match)
   - [Qualification Requirements](#46-qualification-requirements)
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
   - [Triathlon Podium](#64-triathlon-podium)
7. [Settings & Preferences](#7-settings--preferences)
   - [Account Settings](#71-account-settings)
   - [Display Preferences](#72-display-preferences)
   - [Email Notifications](#73-email-notifications)
   - [Dark Mode](#74-dark-mode)
8. [Referee Guide](#8-referee-guide)
   - [Referee Role Overview](#81-referee-role-overview)
   - [Accessing the Referee Dashboard](#82-accessing-the-referee-dashboard)
   - [Managing Matches](#83-managing-matches)
   - [QSO Disqualification](#84-qso-disqualification)
9. [Administrator Guide](#9-administrator-guide)
   - [Admin Authentication](#91-admin-authentication)
   - [Creating an Olympiad](#92-creating-an-olympiad)
   - [Creating Sports](#93-creating-sports)
   - [Creating Matches](#94-creating-matches)
   - [Managing Competitors](#95-managing-competitors)
   - [Managing Teams](#96-managing-teams)
   - [Site Settings](#97-site-settings)
   - [Discord Notifications](#98-discord-notifications)
10. [Deployment & Configuration](#10-deployment--configuration)
    - [Environment Variables](#101-environment-variables)
    - [Local Development](#102-local-development)
    - [Fly.io Deployment](#103-flyio-deployment)
    - [Running Tests](#104-running-tests)

**Appendices:**

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
6. Optionally provide an email address for notifications and password recovery
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
- Personal best records (current season and all-time)
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

**Important:** Unlike the QSO Race (where the first three confirmed contacts lock in the medals), Cool Factor is an ongoing competition that runs until the match ends. Medal standings can change throughout the match as new QSOs are logged. Even if someone currently holds a gold medal, you can knock them out of medal contention by logging a QSO with a higher cool factor. Keep trying until the match closes!

### 3.4 QRP Competitions

Some matches may have a maximum power limit for QRP competitions. Only QSOs at or below the specified power will qualify for that match. Check the match details to see if there's a power restriction.

QRP matches encourage low-power operating and reward efficient stations and skilled operators who can make long-distance contacts with minimal power.

### 3.5 QSO Disqualifications

Occasionally, a referee or the system may disqualify one of your QSOs from medal contention. This can happen for reasons such as:

- Rule violations specific to a sport
- Data anomalies (e.g., malformed park references like "K-11" instead of "K-0011")
- Suspected logging errors

**What happens when a QSO is disqualified:**

- The QSO is removed from medal calculations for that specific sport
- Medals are automatically recomputed
- The disqualification only affects that sport—the same QSO may still count in other sports
- You will see a disqualification indicator on the affected QSO

**Refuting a disqualification:**

If you believe a disqualification was made in error, you can submit a refutation:

1. Navigate to the QSO's disqualification history
2. Click "Refute" and provide your explanation
3. A referee will review your refutation and may requalify the QSO

All disqualification history, including your refutation and any referee responses, is publicly visible for transparency.

---

## 4. Scoring & Medals

### 4.1 Medal Point Values

| Medal | Points |
|-------|--------|
| Gold | 3 points |
| Silver | 2 points |
| Bronze | 1 point |

### 4.2 One Podium Spot Per Competitor

A competitor can only occupy one spot on any podium. This applies to all medal events: match medals, sport standings, and special awards like the Triathlon Podium.

If you have multiple qualifying results (e.g., multiple QSOs in a Cool Factor competition), only your best result counts toward the podium. The remaining spots go to other competitors with the next-best results.

**Example:** If your QSOs rank 1st and 3rd in a Cool Factor competition, you receive the Gold medal. The Silver goes to whoever had the 2nd-best QSO, and Bronze goes to whoever had the 4th-best (since your 3rd-place QSO doesn't count—you already have Gold).

### 4.3 POTA Bonuses

Park contacts earn bonus points in addition to any medals:

| Scenario | Bonus Points |
|----------|--------------|
| Park-to-Park (both stations at parks) | +2 points |
| Target is a park OR you're at a park | +1 point |
| No park involvement | +0 points |

#### POTA Activation Requirements

For **activate mode** competitions targeting parks, a valid activation requires **10 or more confirmed QSOs** from the same park on the same UTC day. This matches POTA's official activation rules.

**Important:** Once a day qualifies as a valid activation (10+ QSOs), **all QSOs from that day count** toward medals—including the first 9. The 10-QSO threshold determines whether the day is valid, not which individual QSOs count.

**Example:**
- Monday: 8 QSOs from K-0001 → Day does NOT qualify (fewer than 10)
- Tuesday: 15 QSOs from K-0001 → Day qualifies, ALL 15 QSOs count toward medals

This requirement only applies to activate mode. Hunters (work mode) have no minimum QSO requirement.

### 4.4 Scoring Examples

**Best possible score (Park-to-Park with double gold):**
- Gold in QSO Race: 3 points
- Gold in Cool Factor: 3 points
- Park-to-Park bonus: +2 points
- **Total: 8 points**

**Strong performance with park bonus:**
- Gold in QSO Race: 3 points
- Silver in Cool Factor: 2 points
- Hunting a park (or activating): +1 point
- **Total: 6 points**

**Solid showing without park involvement:**
- Silver in QSO Race: 2 points
- Bronze in Cool Factor: 1 point
- No park bonus: +0 points
- **Total: 3 points**

**Efficiency specialist (low power, long distance):**
- No QSO Race medal: 0 points
- Gold in Cool Factor: 3 points
- Activated from a park: +1 point
- **Total: 4 points**

**Speed demon (first to make contact):**
- Gold in QSO Race: 3 points
- No Cool Factor medal (high power): 0 points
- Hunted a park: +1 point
- **Total: 4 points**

### 4.5 Maximum Points Per Match

- **Combined pools:** 8 points maximum (Gold QSO Race + Gold Cool Factor + Park-to-Park bonus)
- **Separate pools:** 16 points maximum (8 points per role when both hunting and activating)

### 4.6 Qualification Requirements

The Olympiad administrator can set a minimum QSO threshold for medal eligibility. This prevents casual participants from claiming medals over competitors who are actively engaged in the competition.

**Why this matters:** Without a qualification threshold, someone could make a single exceptional QRP contact (e.g., coast-to-coast on 100mW) and win the Cool Factor gold medal despite not participating in any other matches. The qualification requirement ensures medals go to competitors who are genuinely competing throughout the Olympiad.

**How it works:**
- You must have the minimum number of confirmed QSOs across all matches to be medal-eligible
- You'll still appear in standings before reaching the threshold
- Once you meet the minimum, you become eligible for medals in all matches
- The threshold is set at the Olympiad level and applies to all sports

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

- Longest distance QSO
- Highest Cool Factor score
- Longest distance QRP contact
- Records by mode (CW, SSB, FT8, etc.)

View current records at `/records`.

#### Honorable Mentions

When a confirmed QSO exists that surpasses the current world record but wasn't made during competition, it appears as an "Honorable Mention" below the records table. A QSO is considered "outside competition" if it was either:

- Made outside the active Olympiad date range, OR
- Made during the Olympiad but doesn't match any active match target

This feature recognizes exceptional QSOs while maintaining the integrity of competition records. Think of it as a humbling reminder: someone out there did better when it didn't count.

### 6.2 Personal Bests

Your competitor profile tracks your personal best performances:

- **Current Season:** Best performances during the active Olympiad
- **All-Time:** Your best performances across all Olympiads

These are displayed on your dashboard alongside your medal count and total points, allowing you to measure your improvement over time.

### 6.3 Medal Standings

The Medal Standings page (`/medals`) displays all competitors ranked by their total medal count. The leaderboard shows:

- Gold, Silver, and Bronze medal counts
- Total medals earned
- Total points accumulated

This provides a quick overview of who's leading the competition across all sports.

### 6.4 Triathlon Podium

The Triathlon Podium is a unique recognition exclusive to Ham Radio Olympics that highlights QSOs excelling across all three dimensions of amateur radio competition.

#### What is the Triathlon?

Just as Olympic triathlons test athletes across swimming, cycling, and running, the Ham Radio Triathlon measures excellence across three distinct "events":

1. **Distance** — How far did your signal travel?
2. **Cool Factor** — How efficiently did you operate? (distance ÷ power)
3. **POTA** — Did you involve parks in your contact?

A single exceptional QSO that scores well in all three events can earn a spot on the Triathlon Podium.

#### How Scoring Works

Each qualifying QSO receives a **Triathlon Score** (maximum 300 points):

| Component | Calculation | Max Points |
|-----------|-------------|------------|
| Distance Percentile | Your QSO's rank among all QSOs by distance | 100 |
| Cool Factor Percentile | Your QSO's rank among all QSOs by efficiency | 100 |
| POTA Bonus | P2P (both at parks) = 100, single park = 50 | 100 |

**Example:** A QSO that ranks in the 90th percentile for distance (90 pts), 85th percentile for Cool Factor (85 pts), and is Park-to-Park (100 pts) would score **275 points**.

#### Qualification Requirements

To qualify for the Triathlon Podium, a QSO must:

- Have a positive distance (distance_km > 0)
- Have positive transmit power recorded (tx_power_w > 0)
- Involve at least one POTA park (MY_SIG_INFO or SIG_INFO present)
- Be confirmed (via QRZ or LoTW)


#### Why Triathlon is Unique

The Triathlon Podium recognizes a different kind of excellence than traditional medals:

- **QSO Race** rewards speed — being first to make contact
- **Cool Factor** rewards efficiency — maximum distance with minimum power
- **Triathlon** rewards the complete operator — someone who achieved distance, efficiency, AND park involvement in a single contact

This means a QSO that might not medal in any individual event could still earn Triathlon recognition by performing well across all three dimensions. It celebrates well-rounded operating rather than specialization in a single metric.

The top 3 Triathlon QSOs are displayed at the top of the Records page (`/records`) with a detailed breakdown showing how each QSO scored in each dimension.

---

## 7. Settings & Preferences

Access your settings from the dropdown menu in the navigation bar.

### 7.1 Account Settings

Manage your account information:

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

### 7.2 Display Preferences

Customize how information is displayed throughout the site:

**Distance Unit**
- Choose between Kilometers (km) or Miles (mi)
- All distances on the site will be converted to your preferred unit

**Time Display**
- **UTC:** All times shown in Coordinated Universal Time
- **Local Time:** Times are converted to local time based on the QSO's grid square location

### 7.3 Email Notifications

If you've added and verified an email address, you can receive notifications for:

- **Medal notifications:** When you earn a new medal
- **Weekly match digest:** Upcoming matches in sports you've entered
- **Record notifications:** When you set a new world record

You can enable or disable each notification type individually, or turn off all notifications at once.

### 7.4 Dark Mode

Use the moon/sun icon in the navigation bar to toggle between light and dark mode. The site also respects your system's color scheme preference.

---

## 8. Referee Guide

### 8.1 Referee Role Overview

Referees are competitors with elevated privileges for specific sports. The referee role is granted by an administrator and allows you to manage matches and view detailed competitor information within your assigned sport(s).

### 8.2 Accessing the Referee Dashboard

Once granted referee status, you can access the referee dashboard at `/referee`. This provides an overview of your assigned sports and any pending tasks.

### 8.3 Managing Matches

As a referee for a sport, you can:

- View all matches in your assigned sport(s)
- Create new matches within your assigned sport(s)
- Edit existing matches (dates, targets, power limits)
- Delete matches if needed
- Access detailed competitor information and QSO logs
- Monitor match progress and standings
- Disqualify competitors from your sport(s)
- Disqualify or requalify individual QSOs

### 8.4 QSO Disqualification

Referees can disqualify individual QSOs that violate competition rules. This is different from disqualifying a competitor entirely—a QSO disqualification only affects that specific contact for that specific sport.

**Key points about QSO disqualification:**

- **Sport-specific:** A QSO can be disqualified in one sport but remain valid in another
- **Requires reason:** You must provide a detailed reason (minimum 20 characters) explaining why the QSO is being disqualified
- **Automatic medal update:** Medals are automatically recomputed after a disqualification
- **Audit trail:** All disqualification actions are logged for transparency

**To disqualify a QSO:**

1. Navigate to the competitor's profile or the match standings
2. Find the QSO in question
3. Click the disqualify option and provide your reason
4. The system will remove the QSO from medal contention and recompute standings

**Competitor refutation:** After a QSO is disqualified, the competitor can submit a refutation explaining why they believe the QSO should be reinstated. Refutations appear in the disqualification history for review.

**Requalifying a QSO:** If a disqualification was made in error, or after reviewing a competitor's refutation, referees can requalify the QSO. This restores the QSO to medal contention and triggers another medal recomputation.

**Automatic disqualification:** The system may automatically disqualify QSOs with data anomalies, such as malformed POTA park references (e.g., "K-11" instead of "K-0011"). Competitors can refute these auto-disqualifications, and referees can requalify them after review.

---

## 9. Administrator Guide

### 9.1 Admin Authentication

Administrator access can be granted in two ways:

1. **Admin User Account:** A competitor account with admin privileges. Admins can grant admin status to other users via the admin dashboard.
2. **API Key Header:** For programmatic access, include the `X-Admin-Key` header with requests. This key is configured as an environment variable (`ADMIN_KEY`) during deployment.

To access the admin dashboard, log in with an admin account and navigate to `/admin`.

### 9.2 Creating an Olympiad

An Olympiad is the top-level container for a competition season. To create one:

1. Access the admin dashboard at `/admin`
2. Click "Create Olympiad"
3. Provide name, start date, end date, and qualifying QSO minimum
4. After creation, activate the Olympiad to make it current

Only one Olympiad can be active at a time.

### 9.3 Creating Sports

Sports define categories of competition. Configuration options include:

| Setting | Options / Description |
|---------|----------------------|
| Target Type | continent, country, park, call, grid, any |
| Interval | daily, weekly, bi-weekly, monthly, quarterly, annually |
| Work Enabled | true/false — allows hunting (working stations) |
| Activate Enabled | true/false — allows activating (being the target) |
| Separate Pools | true/false — if true, hunters and activators compete separately |

### 9.4 Creating Matches

Matches are timed events within a sport. Each match requires:

- **Start Date/Time:** When the match begins (UTC recommended)
- **End Date/Time:** When the match ends
- **Target Value:** The specific target (e.g., "EU" for continent, "K-1234" for park)

Optional match settings:

- **Allowed Modes:** Override sport-level mode restrictions for this match only (e.g., "CW,SSB")
- **Max Power (watts):** Limit the match to QRP contacts — only QSOs at or below this power will qualify

### 9.5 Managing Competitors

Administrators can:

- View all registered competitors at `/admin/competitors`
- Enable or disable competitor accounts
- Grant or revoke admin privileges
- Grant or revoke referee roles
- Assign referees to specific sports
- Reset competitor passwords
- Remove competitors if necessary

### 9.6 Managing Teams

Administrators have full team management capabilities:

- View all teams at `/admin/teams`
- Create teams directly
- Add or remove team members
- Transfer team captaincy
- Delete teams if necessary

### 9.7 Site Settings

Administrators can customize the site appearance:

- **Theme:** Choose from multiple visual themes
- **Site Name:** Customize the site title
- **Tagline:** Set a custom tagline displayed in the header
- **QRZ Credentials:** Configure QRZ XML API for callsign lookups

### 9.8 Discord Notifications

Ham Radio Olympics can send automatic notifications to a Discord channel when key events occur. This keeps your club informed about competition activity in real-time.

#### What Gets Posted to Discord

| Event | When It Triggers | Deduplication |
|-------|------------------|---------------|
| **New World Records** | Someone sets a new distance or cool factor record | Once per record value |
| **Medal Awards** | Competitors earn gold, silver, or bronze medals | Once per medal |
| **New Signups** | A new competitor joins | Once per callsign |
| **Match Reminders** | 7 days, 1 day, and day-of for upcoming matches | Once per match |
| **POTA Activity Summary** | Every 30 minutes if parks have active spots | Once per 30 minutes |

#### Message Formats

Each notification type has a distinct appearance in Discord:

| Event | Title | Example Message | Color |
|-------|-------|-----------------|-------|
| **World Record** | 🥇 New World Record! 📏 | **KJ5IRF** set a new distance record! Distance: 15,234 km | Gold |
| **Medal (Gold)** | 🥇 Gold Medal Awarded! | **KJ5IRF** earned gold in QSO Race | Gold |
| **Medal (Silver)** | 🥈 Silver Medal Awarded! | **W1AW** earned silver in Cool Factor | Silver |
| **Medal (Bronze)** | 🥉 Bronze Medal Awarded! | **N0CALL** earned bronze in QSO Race | Bronze |
| **New Signup** | 👋 New Competitor Joined! | Welcome **KJ5IRF** to Ham Radio Olympics! | Green |
| **Match (Today)** | 🚀 Match Starting Today! | **DX Challenge** targeting **EU** starts today | Red |
| **Match (Tomorrow)** | ⏰ Match Starting Tomorrow! | **POTA Championship** targeting **K-0001** starts tomorrow | Orange |
| **Match (7 days)** | 📅 Match in 7 Days | **Grid Chase** targeting **FN31** starts in 7 days | Blue |
| **POTA Activity** | 📡 POTA Activity (X spots) | • POTA Championship: 5 active spots | Green |

#### Deduplication

All Discord notifications are deduplicated to prevent spam:
- **Records and medals** are only announced once, even if data is recomputed
- **POTA activity** summary is sent once every 30 minutes (batched, not per-spot)
- **Match reminders** are sent once per match at each reminder interval

#### Setting Up Discord Notifications

**Step 1: Create a Webhook in Discord**

1. Open Discord and go to your server
2. Click the server name at the top → **Server Settings**
3. In the left menu, go to **Apps** → **Integrations**
4. Click **Webhooks** → **New Webhook**
5. Configure the webhook:
   - **Name:** Give it a name like "Ham Radio Olympics" (this appears as the sender)
   - **Channel:** Select which channel should receive notifications
   - **Avatar (optional):** Use the Ham Radio Olympics icon: `https://kd5dx.fly.dev/static/icon-512.png`
6. Click **Copy Webhook URL**

**Step 2: Add the Webhook to Ham Radio Olympics**

1. Log in as an administrator
2. Go to **Admin** → **Settings**
3. Find the **Discord Notifications** section
4. Paste the webhook URL
5. Click **Save**
6. Click **Test** to verify — you should see a test message in your Discord channel

#### Troubleshooting

- **Test message not appearing:** Verify the webhook URL starts with `https://discord.com/api/webhooks/`
- **Wrong channel:** Edit the webhook in Discord Server Settings to change the target channel
- **Want to disable:** Click **Clear** in the admin settings to stop Discord notifications

---

## 10. Deployment & Configuration

### 10.1 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ENCRYPTION_KEY` | Secret key for encrypting QRZ API keys at rest | Yes |
| `ADMIN_KEY` | Secret for administrator authentication | Yes |
| `DATABASE_PATH` | Path to SQLite database file | No (default: `ham_olympics.db`) |
| `PORT` | Server port | No (default: `8000`) |

### 10.2 Local Development

To run Ham Radio Olympics locally for development or testing:

1. Clone the repository from GitHub
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables (`ENCRYPTION_KEY`, `ADMIN_KEY`)
4. Run: `python -m uvicorn main:app --reload`
5. Open http://localhost:8000 in your browser

### 10.3 Fly.io Deployment

Ham Radio Olympics includes a Dockerfile and `fly.toml` for easy deployment to Fly.io:

1. Install flyctl: `curl -L https://fly.io/install.sh | sh`
2. Login: `fly auth login`
3. Create app: `fly launch --name ham-radio-olympics`
4. Create persistent volume: `fly volumes create ham_olympics_data --size 1 --region iad`
5. Set secrets: `fly secrets set ENCRYPTION_KEY="..." ADMIN_KEY="..."`
6. Deploy: `fly deploy`

### 10.4 Running Tests

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
| POST | `/referee/sport/{id}/qso/{qso_id}/disqualify` | Disqualify a QSO |
| POST | `/referee/sport/{id}/qso/{qso_id}/requalify` | Requalify a disqualified QSO |
| GET | `/qso/{qso_id}/disqualifications` | View QSO disqualification history (public) |

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

*— End of Document —*
