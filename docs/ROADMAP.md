# Ham Radio Olympics - Roadmap & Future Features

This document captures potential improvements and missing features identified during the v1.0.0 release review.

---

## Competition Features

### 1. Certificates/Awards
No certificate generation for medal winners. Most ham contests award downloadable PDF certificates.

### 2. Power Categories
Max power per match exists, but no explicit QRP/Low/High power classes with separate leaderboards.

### 3. Band-Specific Competitions
Sports filter by mode but not by band. A "10 Meter Challenge" or "6 Meter Magic" sport would need band filtering.

### 4. Claimed vs Confirmed Scores
Only confirmed QSOs count. Some competitors may want to see their "claimed" score while waiting for confirmations.

### 5. Multi-Operator Categories
No single-op vs multi-op distinction. Field Day style operations would need this.

### 6. Club Competition
Teams exist, but clubs are different (larger, more formal). Many contests have club aggregates.

---

## UI/UX

### 7. Real-time Updates
No WebSocket/SSE. Leaderboards require refresh to see changes.

### 8. In-App Notifications
Email notifications work, but no notification bell/center in the UI.

### 9. Activity Feed
No "recent activity" showing medals awarded, records broken, new competitors joining.

### 10. Search
No global search across competitors, teams, sports.

### 11. Comparison Tools
Can't compare two competitors side-by-side.

### 12. Visualizations
No charts/graphs. Progress over time, QSOs by hour, geographic heat maps would be valuable.

### 13. Calendar View
Matches have dates but no calendar visualization of upcoming/past matches.

### 14. Competitor Profiles
Limited to name only. No bio, photo, location description, equipment list.

---

## Social/Community

### 15. Discussion/Comments
No way to discuss matches or congratulate winners.

### 16. Following/Friends
Can't follow other competitors to track their progress.

### 17. Achievement Sharing
No "Share to Twitter/Mastodon" or embed codes for achievements.

---

## Ham Radio Specific

### 18. Other Outdoor Programs
Only POTA. What about SOTA (Summits on the Air), WWFF (World Wide Flora & Fauna), IOTA (Islands)?

### 19. More Log Sources
Only QRZ and LoTW. Could add eQSL, ClubLog, direct ADIF upload.

### 20. Propagation Display
No solar indices (SFI, A/K index), band conditions, or propagation predictions.

### 21. DX Cluster Integration
POTA spots work, but no general DX cluster spots for DX Challenge sports.

### 22. Awards Tracking
No tracking of progress toward external awards (DXCC, WAS, VUCC).

### 23. Equipment Profiles
No way to log what rig/antenna was used. Cool Factor would be more impressive knowing it was a wire antenna vs a beam.

---

## Technical

### 24. Two-Factor Authentication
Listed as "2FA ready" but not implemented.

### 25. Webhooks
No outbound webhooks for third-party integrations (Discord bots, custom dashboards).

### 26. Localization
English only. Ham radio is global.

### 27. Status Page
No system status or uptime indicator.

---

## Quick Wins (Low Effort, High Value)

| Feature | Effort | Impact |
|---------|--------|--------|
| Certificates | Low | High - PDF generation already exists |
| Activity feed | Low | Medium - Query recent medals/records |
| Global search | Low | Medium - Simple LIKE queries |
| Band filtering | Low | High - Add `allowed_bands` like `allowed_modes` |
| Direct ADIF upload | Medium | High - Bypass QRZ for manual uploads |

---

*Generated during v1.0.0 release review - January 2026*
