/**
 * Ham Radio Olympics - Production Service Worker
 * Version: 2.0.0
 *
 * Caching Strategies:
 * - Static assets (icons, CSS): Cache-first with long TTL
 * - HTML pages: Network-first with cache fallback
 * - API requests: Network-only (with background sync for offline)
 * - User guide: Stale-while-revalidate
 */

const SW_VERSION = '2.1.2';
const CACHE_PREFIX = 'hro-';
const STATIC_CACHE = `${CACHE_PREFIX}static-v${SW_VERSION}`;
const PAGES_CACHE = `${CACHE_PREFIX}pages-v${SW_VERSION}`;
const RUNTIME_CACHE = `${CACHE_PREFIX}runtime-v${SW_VERSION}`;

// Assets to precache on install
const PRECACHE_ASSETS = [
  '/',
  '/static/manifest.json',
  '/static/offline.html',
  '/static/favicon.ico',
  '/static/favicon.svg',
  '/static/favicon-16x16.png',
  '/static/favicon-32x32.png',
  '/static/apple-touch-icon.png',
  '/static/icon-192.png',
  '/static/icon-512.png',
  '/static/themes/olympics.css',
  '/static/themes/coolcontest.css',
  '/static/themes/midnight.css',
  '/static/themes/neon.css',
  '/static/user_guide.html'
];

// Pages to cache on first visit (network-first)
const CACHE_PAGES = [
  '/dashboard',
  '/olympiad',
  '/olympiad/sports',
  '/records',
  '/medals',
  '/teams'
];

// Never cache these paths
const NEVER_CACHE = [
  '/sync',
  '/login',
  '/logout',
  '/signup',
  '/api/',
  '/admin',
  '/referee',
  '/export'
];

// Maximum items in runtime cache
const MAX_RUNTIME_CACHE_ITEMS = 50;

// Network timeout before falling back to cache (ms)
// 8 seconds to allow for Fly.io cold starts (~1-2s) plus network latency
const NETWORK_TIMEOUT = 8000;

/**
 * Install Event - Precache static assets
 */
self.addEventListener('install', (event) => {
  console.log(`[SW ${SW_VERSION}] Installing...`);

  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log(`[SW ${SW_VERSION}] Precaching assets`);
        return cache.addAll(PRECACHE_ASSETS);
      })
      .then(() => {
        console.log(`[SW ${SW_VERSION}] Skip waiting`);
        return self.skipWaiting();
      })
      .catch((error) => {
        console.error(`[SW ${SW_VERSION}] Precache failed:`, error);
      })
  );
});

/**
 * Activate Event - Clean old caches and claim clients
 */
self.addEventListener('activate', (event) => {
  console.log(`[SW ${SW_VERSION}] Activating...`);

  event.waitUntil(
    Promise.all([
      // Clean old caches
      caches.keys().then((cacheNames) => {
        return Promise.all(
          cacheNames
            .filter((name) => name.startsWith(CACHE_PREFIX))
            .filter((name) =>
              name !== STATIC_CACHE &&
              name !== PAGES_CACHE &&
              name !== RUNTIME_CACHE
            )
            .map((name) => {
              console.log(`[SW ${SW_VERSION}] Deleting old cache: ${name}`);
              return caches.delete(name);
            })
        );
      }),
      // Claim all clients
      self.clients.claim()
    ]).then(() => {
      console.log(`[SW ${SW_VERSION}] Activated and controlling`);
      // Notify clients of update
      self.clients.matchAll().then((clients) => {
        clients.forEach((client) => {
          client.postMessage({
            type: 'SW_UPDATED',
            version: SW_VERSION
          });
        });
      });
    })
  );
});

/**
 * Fetch Event - Route requests to appropriate strategy
 */
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Only handle same-origin requests
  if (url.origin !== self.location.origin) {
    return;
  }

  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }

  // Skip never-cache paths
  if (NEVER_CACHE.some((path) => url.pathname.startsWith(path))) {
    return;
  }

  // Route to appropriate strategy
  if (isStaticAsset(url.pathname)) {
    event.respondWith(cacheFirst(request, STATIC_CACHE));
  } else if (isNavigationRequest(request)) {
    event.respondWith(networkFirstWithOffline(request));
  } else {
    event.respondWith(staleWhileRevalidate(request, RUNTIME_CACHE));
  }
});

/**
 * Check if request is for a static asset
 */
function isStaticAsset(pathname) {
  return pathname.startsWith('/static/') ||
         pathname.endsWith('.css') ||
         pathname.endsWith('.js') ||
         pathname.endsWith('.png') ||
         pathname.endsWith('.jpg') ||
         pathname.endsWith('.svg') ||
         pathname.endsWith('.ico') ||
         pathname.endsWith('.woff2');
}

/**
 * Check if request is a navigation (HTML page) request
 */
function isNavigationRequest(request) {
  return request.mode === 'navigate' ||
         request.headers.get('Accept')?.includes('text/html');
}

/**
 * Cache-First Strategy
 * Best for: Static assets that rarely change
 */
async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);

  if (cached) {
    return cached;
  }

  try {
    const response = await fetch(request);
    if (response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    console.error(`[SW] Cache-first failed for ${request.url}:`, error);
    throw error;
  }
}

/**
 * Network-First with Offline Fallback
 * Best for: HTML pages
 */
async function networkFirstWithOffline(request) {
  const cache = await caches.open(PAGES_CACHE);

  try {
    // Try network with timeout
    const response = await fetchWithTimeout(request, NETWORK_TIMEOUT);

    if (response.ok) {
      // Cache successful responses
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    console.log(`[SW] Network failed for ${request.url}, checking cache`);

    // Try cache
    const cached = await cache.match(request);
    if (cached) {
      return cached;
    }

    // Fallback to offline page for navigation requests
    const offlinePage = await caches.match('/static/offline.html');
    if (offlinePage) {
      return offlinePage;
    }

    // Last resort: return cached home page
    const homePage = await caches.match('/');
    if (homePage) {
      return homePage;
    }

    throw error;
  }
}

/**
 * Stale-While-Revalidate Strategy
 * Best for: Content that updates but stale is acceptable
 */
async function staleWhileRevalidate(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);

  // Start network fetch in background
  const networkPromise = fetch(request)
    .then((response) => {
      if (response.ok) {
        cache.put(request, response.clone());
        // Trim cache if too large
        trimCache(cacheName, MAX_RUNTIME_CACHE_ITEMS);
      }
      return response;
    })
    .catch((error) => {
      console.log(`[SW] SWR network failed for ${request.url}`);
      return null;
    });

  // Return cached immediately if available, otherwise wait for network
  return cached || networkPromise;
}

/**
 * Fetch with timeout
 */
function fetchWithTimeout(request, timeout) {
  return new Promise((resolve, reject) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      controller.abort();
      reject(new Error('Network timeout'));
    }, timeout);

    fetch(request, { signal: controller.signal })
      .then((response) => {
        clearTimeout(timeoutId);
        resolve(response);
      })
      .catch((error) => {
        clearTimeout(timeoutId);
        reject(error);
      });
  });
}

/**
 * Trim cache to maximum number of items
 */
async function trimCache(cacheName, maxItems) {
  const cache = await caches.open(cacheName);
  const keys = await cache.keys();

  if (keys.length > maxItems) {
    // Delete oldest entries (first in list)
    const deleteCount = keys.length - maxItems;
    for (let i = 0; i < deleteCount; i++) {
      await cache.delete(keys[i]);
    }
    console.log(`[SW] Trimmed ${deleteCount} items from ${cacheName}`);
  }
}

/**
 * Background Sync - Queue failed sync requests
 */
self.addEventListener('sync', (event) => {
  console.log(`[SW] Background sync: ${event.tag}`);

  if (event.tag === 'sync-qsos') {
    event.waitUntil(syncQsos());
  }
});

/**
 * Process queued sync requests
 */
async function syncQsos() {
  try {
    // Get queued sync requests from IndexedDB
    const queue = await getQueuedSyncs();

    for (const item of queue) {
      try {
        const response = await fetch(item.url, {
          method: 'POST',
          headers: item.headers,
          body: item.body
        });

        if (response.ok) {
          await removeFromQueue(item.id);
          // Notify client of successful sync
          self.clients.matchAll().then((clients) => {
            clients.forEach((client) => {
              client.postMessage({
                type: 'SYNC_COMPLETE',
                url: item.url
              });
            });
          });
        }
      } catch (error) {
        console.error(`[SW] Queued sync failed:`, error);
      }
    }
  } catch (error) {
    console.error(`[SW] Background sync error:`, error);
  }
}

/**
 * IndexedDB helpers for sync queue
 */
const DB_NAME = 'hro-sync-queue';
const STORE_NAME = 'pending-syncs';

function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id', autoIncrement: true });
      }
    };
  });
}

async function getQueuedSyncs() {
  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const request = store.getAll();

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result || []);
    });
  } catch (error) {
    console.error('[SW] Failed to get queued syncs:', error);
    return [];
  }
}

async function removeFromQueue(id) {
  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const request = store.delete(id);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve();
    });
  } catch (error) {
    console.error('[SW] Failed to remove from queue:', error);
  }
}

/**
 * Push Notification handler
 */
self.addEventListener('push', (event) => {
  if (!event.data) return;

  try {
    const data = event.data.json();

    const options = {
      body: data.body || 'New update from Ham Radio Olympics',
      icon: '/static/icon-192.png',
      badge: '/static/icon-96.png',
      vibrate: [100, 50, 100],
      data: {
        url: data.url || '/',
        timestamp: Date.now()
      },
      actions: data.actions || [
        { action: 'view', title: 'View' },
        { action: 'dismiss', title: 'Dismiss' }
      ],
      tag: data.tag || 'hro-notification',
      renotify: data.renotify || false
    };

    event.waitUntil(
      self.registration.showNotification(
        data.title || 'Ham Radio Olympics',
        options
      )
    );
  } catch (error) {
    console.error('[SW] Push notification error:', error);
  }
});

/**
 * Notification click handler
 */
self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  if (event.action === 'dismiss') {
    return;
  }

  const url = event.notification.data?.url || '/';

  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true })
      .then((clients) => {
        // Focus existing window if open
        for (const client of clients) {
          if (client.url.includes(self.location.origin) && 'focus' in client) {
            client.navigate(url);
            return client.focus();
          }
        }
        // Open new window
        return self.clients.openWindow(url);
      })
  );
});

/**
 * Message handler for client communication
 */
self.addEventListener('message', (event) => {
  const { type, payload } = event.data || {};

  switch (type) {
    case 'SKIP_WAITING':
      self.skipWaiting();
      break;

    case 'GET_VERSION':
      event.ports[0]?.postMessage({ version: SW_VERSION });
      break;

    case 'QUEUE_SYNC':
      queueSync(payload).then(() => {
        event.ports[0]?.postMessage({ success: true });
      }).catch((error) => {
        event.ports[0]?.postMessage({ success: false, error: error.message });
      });
      break;

    case 'CLEAR_CACHES':
      clearAllCaches().then(() => {
        event.ports[0]?.postMessage({ success: true });
      });
      break;
  }
});

/**
 * Queue a sync request for background processing
 */
async function queueSync(payload) {
  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const request = store.add({
        url: payload.url,
        headers: payload.headers,
        body: payload.body,
        timestamp: Date.now()
      });

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        // Register for background sync
        if ('sync' in self.registration) {
          self.registration.sync.register('sync-qsos');
        }
        resolve();
      };
    });
  } catch (error) {
    console.error('[SW] Failed to queue sync:', error);
    throw error;
  }
}

/**
 * Clear all app caches
 */
async function clearAllCaches() {
  const cacheNames = await caches.keys();
  await Promise.all(
    cacheNames
      .filter((name) => name.startsWith(CACHE_PREFIX))
      .map((name) => caches.delete(name))
  );
  console.log('[SW] All caches cleared');
}

console.log(`[SW ${SW_VERSION}] Loaded`);
