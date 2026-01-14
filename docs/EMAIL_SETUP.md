# Email Setup Guide

This guide covers setting up email for the Ham Radio Olympics application. The app supports three email backends: **Console** (development), **SMTP** (self-hosted), and **Resend** (recommended for production).

## Table of Contents

1. [Overview](#overview)
2. [Environment Variables](#environment-variables)
3. [Backend Options](#backend-options)
   - [Console (Development)](#console-development)
   - [SMTP](#smtp)
   - [Resend (Recommended)](#resend-recommended)
4. [Resend Setup Step-by-Step](#resend-setup-step-by-step)
5. [DNS Configuration](#dns-configuration)
6. [Testing Your Setup](#testing-your-setup)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The application sends the following types of emails:

| Email Type | Trigger | Description |
|------------|---------|-------------|
| Welcome | User signup | Welcomes new users |
| Email Verification | Signup or settings change | Confirms email ownership |
| Password Reset | Forgot password request | Secure password reset link |
| Medal Notification | Admin trigger or cron | Notifies users of new medals |
| Weekly Match Digest | Admin trigger or cron | Lists upcoming matches |
| Record Notification | New world record | Congratulates record holders |

---

## Environment Variables

All email configuration is done through environment variables. For Fly.io deployments, use `fly secrets set`.

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `EMAIL_BACKEND` | Which backend to use | `resend`, `smtp`, or `console` |
| `EMAIL_FROM` | Sender email address | `noreply@yourdomain.com` |

### Backend-Specific Variables

#### For Resend:
| Variable | Description | Example |
|----------|-------------|---------|
| `RESEND_API_KEY` | Your Resend API key | `re_123abc...` |

#### For SMTP:
| Variable | Description | Example |
|----------|-------------|---------|
| `SMTP_HOST` | SMTP server hostname | `smtp.gmail.com` |
| `SMTP_PORT` | SMTP server port | `587` |
| `SMTP_USERNAME` | SMTP authentication username | `user@gmail.com` |
| `SMTP_PASSWORD` | SMTP authentication password | `app-specific-password` |
| `SMTP_USE_TLS` | Enable TLS encryption | `true` |

---

## Backend Options

### Console (Development)

The console backend prints emails to stdout instead of sending them. Useful for local development and testing.

```bash
# Local development
export EMAIL_BACKEND=console
export EMAIL_FROM=test@localhost

# Fly.io
fly secrets set EMAIL_BACKEND=console EMAIL_FROM=test@localhost
```

Emails will appear in the application logs:
```
============================================================
EMAIL (Console Backend)
============================================================
To: user@example.com
Subject: Welcome to Ham Radio Olympics!
------------------------------------------------------------
[email body here]
============================================================
```

### SMTP

Use any SMTP server (Gmail, SendGrid, Mailgun, self-hosted, etc.).

```bash
# Example: Gmail SMTP
fly secrets set \
  EMAIL_BACKEND=smtp \
  EMAIL_FROM=yourapp@gmail.com \
  SMTP_HOST=smtp.gmail.com \
  SMTP_PORT=587 \
  SMTP_USERNAME=yourapp@gmail.com \
  SMTP_PASSWORD=your-app-specific-password \
  SMTP_USE_TLS=true
```

**Gmail Notes:**
- Enable 2FA on your Google account
- Generate an App Password at https://myaccount.google.com/apppasswords
- Use the App Password, not your regular password

**SendGrid SMTP:**
```bash
fly secrets set \
  EMAIL_BACKEND=smtp \
  EMAIL_FROM=noreply@yourdomain.com \
  SMTP_HOST=smtp.sendgrid.net \
  SMTP_PORT=587 \
  SMTP_USERNAME=apikey \
  SMTP_PASSWORD=SG.your-api-key \
  SMTP_USE_TLS=true
```

### Resend (Recommended)

[Resend](https://resend.com) is a modern email API with generous free tier (3,000 emails/month) and simple setup.

```bash
fly secrets set \
  EMAIL_BACKEND=resend \
  EMAIL_FROM=noreply@yourdomain.com \
  RESEND_API_KEY=re_your_api_key
```

---

## Resend Setup Step-by-Step

### Step 1: Create Resend Account

1. Go to https://resend.com
2. Sign up with email or GitHub
3. Verify your email address

### Step 2: Get API Key

1. In Resend dashboard, go to **API Keys**
2. Click **Create API Key**
3. Name it (e.g., "Ham Radio Olympics Production")
4. Select permissions: **Sending access** is sufficient
5. Copy the key (starts with `re_`)

```bash
fly secrets set RESEND_API_KEY=re_your_api_key_here
```

### Step 3: Add Your Domain

**Option A: Use Resend's Test Domain (Limited)**

For testing only, you can send to your own email without domain setup:
```bash
fly secrets set EMAIL_FROM=onboarding@resend.dev
```
*Limitation: Can only send to the email address on your Resend account.*

**Option B: Add Custom Domain (Recommended)**

1. In Resend dashboard, go to **Domains**
2. Click **Add Domain**
3. Enter your domain (e.g., `yourdomain.com`)
4. Choose a subdomain for sending (recommended: `resend.yourdomain.com` or `mail.yourdomain.com`)

### Step 4: Configure DNS Records

Resend will show you DNS records to add. You need:

| Type | Name | Value | Purpose |
|------|------|-------|---------|
| TXT | `resend._domainkey.yourdomain.com` | `p=MIGf...` | DKIM signing |
| TXT | `yourdomain.com` or `_dmarc.yourdomain.com` | `v=DMARC1; p=none;` | DMARC policy |
| MX | `send.yourdomain.com` (if using subdomain) | `feedback-smtp.us-east-1.amazonses.com` | Bounce handling |

**Example DNS Setup (Cloudflare):**

1. Log into Cloudflare dashboard
2. Select your domain
3. Go to DNS > Records
4. Add each record Resend shows you:

```
Type: TXT
Name: resend._domainkey
Content: p=MIGfMA0GCSqGSIb3DQEBAQUAA4GN... (from Resend)
TTL: Auto
Proxy: DNS only (gray cloud)
```

### Step 5: Verify Domain

1. After adding DNS records, wait 5-60 minutes for propagation
2. In Resend dashboard, click **Verify** on your domain
3. Status should change to **Verified**

### Step 6: Set EMAIL_FROM

Use an address on your verified domain:

```bash
# If you verified "resend.yourdomain.com":
fly secrets set EMAIL_FROM=noreply@resend.yourdomain.com

# If you verified the root domain:
fly secrets set EMAIL_FROM=noreply@yourdomain.com
```

**Important:** The EMAIL_FROM domain must match your verified domain exactly.

### Step 7: Set Backend

```bash
fly secrets set EMAIL_BACKEND=resend
```

---

## DNS Configuration

### Required Records for Resend

When you add a domain to Resend, you'll get specific records. Here's what each does:

#### DKIM Record (Required)
```
Type: TXT
Name: resend._domainkey.yourdomain.com
Value: p=MIGf... (provided by Resend)
```
DKIM cryptographically signs your emails to prove they're from you.

#### SPF Record (Recommended)
If you already have an SPF record, add Resend's include:
```
v=spf1 include:amazonses.com ~all
```

If you don't have one:
```
Type: TXT
Name: yourdomain.com
Value: v=spf1 include:amazonses.com ~all
```

#### DMARC Record (Recommended)
```
Type: TXT
Name: _dmarc.yourdomain.com
Value: v=DMARC1; p=none; rua=mailto:dmarc@yourdomain.com
```

### Checking DNS Propagation

Use these tools to verify your records:

```bash
# Check DKIM
dig TXT resend._domainkey.yourdomain.com

# Check SPF
dig TXT yourdomain.com

# Online tools
# - https://mxtoolbox.com/
# - https://dnschecker.org/
```

---

## Testing Your Setup

### Method 1: Password Reset Flow

1. Go to your app's login page
2. Click "Forgot Password"
3. Enter a valid email address
4. Check for the email

### Method 2: Send Test Email via SSH

```bash
# Connect to your Fly.io app
fly ssh console -a your-app-name

# Run Python test
python3 -c "
import asyncio
import sys
sys.path.insert(0, '/app')
from email_service import send_email

async def test():
    success = await send_email(
        to='your-email@example.com',
        subject='Test Email',
        body='This is a test email from Ham Radio Olympics.'
    )
    print('Success!' if success else 'Failed!')

asyncio.run(test())
"
```

### Method 3: Check Logs

```bash
# View recent logs
fly logs -a your-app-name

# Look for email-related entries
fly logs -a your-app-name | grep -i email
```

---

## Troubleshooting

### Error: "Domain is not verified"

**Cause:** EMAIL_FROM uses a domain not verified in Resend.

**Fix:**
1. Check your verified domains in Resend dashboard
2. Make sure EMAIL_FROM matches exactly
3. If using subdomain `resend.yourdomain.com`, use `noreply@resend.yourdomain.com`

```bash
# Check current setting
fly secrets list | grep EMAIL

# Update to match verified domain
fly secrets set EMAIL_FROM=noreply@your-verified-domain.com
```

### Error: "You can only send testing emails to your own email"

**Cause:** Using Resend without a verified domain.

**Fix:** Add and verify your domain in Resend, or test with the email address on your Resend account.

### Error: "Too many requests" / Rate Limiting

**Cause:** Resend free tier allows 2 requests/second.

**Fix:** Add delays between emails in bulk operations:

```python
import asyncio

for email in emails_to_send:
    await send_email(...)
    await asyncio.sleep(1)  # Wait 1 second between sends
```

### Emails Not Arriving

1. **Check spam folder** - New domains often trigger spam filters
2. **Verify DNS records** - Use mxtoolbox.com to check DKIM/SPF/DMARC
3. **Check Resend dashboard** - Look at Logs for delivery status
4. **Check application logs** - Look for send errors

```bash
fly logs -a your-app-name | grep -i "email\|resend\|smtp"
```

### Email Verification Links Not Working

**Cause:** Links use wrong base URL.

**Fix:** Check `APP_BASE_URL` in `email_service.py`:

```python
APP_BASE_URL = "https://your-app.fly.dev"  # Must match your actual URL
```

### SMTP Connection Errors

1. **Check firewall** - Port 587 (TLS) or 465 (SSL) must be open
2. **Verify credentials** - Use app-specific passwords for Gmail
3. **Check TLS setting** - Most providers require `SMTP_USE_TLS=true`

```bash
# Test SMTP connection manually
fly ssh console -a your-app-name
python3 -c "
import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('user@gmail.com', 'app-password')
print('Connected!')
server.quit()
"
```

---

## Quick Reference

### Minimum Resend Setup

```bash
fly secrets set \
  EMAIL_BACKEND=resend \
  EMAIL_FROM=noreply@resend.yourdomain.com \
  RESEND_API_KEY=re_your_api_key
```

### Minimum SMTP Setup

```bash
fly secrets set \
  EMAIL_BACKEND=smtp \
  EMAIL_FROM=noreply@yourdomain.com \
  SMTP_HOST=smtp.example.com \
  SMTP_PORT=587 \
  SMTP_USERNAME=user \
  SMTP_PASSWORD=pass \
  SMTP_USE_TLS=true
```

### Development Setup

```bash
export EMAIL_BACKEND=console
export EMAIL_FROM=test@localhost
```

---

## Code Reference

The email system is implemented in `app/email_service.py`. Key functions:

| Function | Purpose |
|----------|---------|
| `send_email(to, subject, body, html_body)` | Core send function |
| `send_email_verification(callsign, email, url)` | Email verification |
| `send_password_reset_email(callsign, email, url)` | Password reset |
| `send_medal_notification_email(...)` | Medal notifications |
| `send_weekly_match_digest(days_ahead)` | Weekly digest to all users |
| `notify_new_medals()` | Batch notify unnotified medals |

Configuration is loaded from `app/config.py` which reads environment variables.
