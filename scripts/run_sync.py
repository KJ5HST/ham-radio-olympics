#!/usr/bin/env python3
"""
Standalone sync script that can be run as a subprocess.

This script runs the sync independently of the main web server,
ensuring the UI is never blocked by sync operations.

Usage:
    python -m scripts.run_sync
    python -m scripts.run_sync --competitor W1ABC
"""

import asyncio
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db
from sync import sync_all_competitors, sync_competitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Run sync for all competitors or a specific one."""
    init_db()

    if len(sys.argv) > 2 and sys.argv[1] == "--competitor":
        callsign = sys.argv[2].upper()
        logger.info(f"Syncing competitor: {callsign}")
        result = await sync_competitor(callsign)
        logger.info(f"Sync result: {result}")
    else:
        logger.info("Starting full sync of all competitors")
        result = await sync_all_competitors()
        logger.info(f"Sync complete: {result}")

    return result


if __name__ == "__main__":
    asyncio.run(main())
