#!/usr/bin/env python3
"""
Akimbo Club Product Scraper
----------------------------
Scrapes all products from akimboclub.com/collections/akimbo,
generates image & text embeddings using google/siglip-base-patch16-384,
and uploads everything to Supabase.

Usage:
    python scraper.py [--resume] [--limit N]
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import httpx
import numpy as np
import torch
from bs4 import BeautifulSoup
from PIL import Image
from supabase import create_client
from transformers import SiglipModel, SiglipProcessor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPABASE_URL = "https://yqawmzggcgpeyaaynrjk.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4"

BASE_URL = "https://akimboclub.com"
COLLECTION_URL = "https://akimboclub.com/collections/akimbo"
SOURCE_NAME = "scraper-akimboclub"
BRAND_NAME = "Akimbo Club"

CHECKPOINT_FILE = "scraper_checkpoint.json"
PROGRESS_FILE = "scraper_progress.json"
MODEL_NAME = "google/siglip-base-patch16-384"
EMBEDDING_DIM = 768
BATCH_SIZE = 50
STALE_THRESHOLD = 2  # Consecutive runs without being seen before deletion
STAGGER_DELAY = 0.5  # Seconds between embedding API calls

# Concurrency
MAX_CONCURRENT_FETCHES = 5
MAX_CONCURRENT_EMBEDDINGS = 2  # GPU memory — keep low

# Retry
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# Logging
FAILED_LOG_FILE = "scraper_failed_products.jsonl"

# Headers
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scraper")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_id(product_url: str) -> str:
    """Generate a deterministic ID from a product URL."""
    return hashlib.md5(product_url.encode()).hexdigest()


def load_checkpoint() -> dict:
    """Load saved progress so we can resume after interruptions."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"done": [], "failed": []}


def save_checkpoint(checkpoint: dict) -> None:
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def save_progress(stats: dict) -> None:
    with open(PROGRESS_FILE, "w") as f:
        json.dump(stats, f, indent=2, default=str)


def product_changed(scraped: dict, existing: dict) -> bool:
    """
    Check if scraped product data differs from the existing database record.
    Returns True if any tracked field has changed.
    """
    # Compare scalar fields that matter for updates
    compare_fields = [
        "title", "description", "price", "sale", "image_url",
        "additional_images", "category", "size", "brand",
        "gender", "affiliate_url",
    ]
    for field in compare_fields:
        if scraped.get(field) != existing.get(field):
            log.debug(f"  Change detected in '{field}'")
            return True

    # Compare tags (order-independent)
    scraped_tags = set(scraped.get("tags") or [])
    existing_tags = set(existing.get("tags") or [])
    if scraped_tags != existing_tags:
        log.debug("  Change detected in 'tags'")
        return True

    # Compare metadata (JSON string — parse to compare variants, availability, etc.)
    try:
        scraped_meta = json.loads(scraped.get("metadata", "{}"))
        existing_meta = json.loads(existing.get("metadata", "{}"))
        # Compare relevant sub-structures
        for meta_key in ["variants", "sizes", "category"]:
            if scraped_meta.get(meta_key) != existing_meta.get(meta_key):
                log.debug(f"  Change detected in metadata.{meta_key}")
                return True
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    return False


def image_url_changed(scraped: dict, existing: dict) -> bool:
    """Check if the product's main image URL has changed."""
    return scraped.get("image_url") != existing.get("image_url")


def extract_json_ld(soup: BeautifulSoup) -> dict | None:
    """Extract the first JSON-LD block with @type == 'Product'."""
    for script in soup.find_all("script", type="application/ld+json"):
        if script.string:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and data.get("@type") == "Product":
                    return data
                # Some Shopify themes wrap it in @graph or array
                if isinstance(data, dict) and "@graph" in data:
                    for item in data["@graph"]:
                        if isinstance(item, dict) and item.get("@type") == "Product":
                            return item
            except (json.JSONDecodeError, AttributeError):
                continue
    return None


def normalize_price(price_str: str | float | int | None) -> str | None:
    """Convert a price to 'X.XXUSD' format."""
    if price_str is None:
        return None
    try:
        val = float(price_str)
        return f"{val:.2f}USD"
    except (ValueError, TypeError):
        return str(price_str)


# ---------------------------------------------------------------------------
# Product Page Scraper
# ---------------------------------------------------------------------------

class ProductScraper:
    """Parse a single product page from akimboclub.com."""

    def __init__(self, url: str, html: str) -> None:
        self.url = url
        self.soup = BeautifulSoup(html, "lxml")
        self.json_ld = extract_json_ld(self.soup)

    def extract_title(self) -> str:
        """Extract product title from h1 or JSON-LD."""
        if self.json_ld:
            name = self.json_ld.get("name")
            if name:
                return name.strip()
        h1 = self.soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        # Fallback: og:title meta
        og = self.soup.find("meta", property="og:title")
        if og:
            return og.get("content", "").strip()
        return "Unknown Product"

    def extract_description(self) -> str | None:
        """Extract description from JSON-LD or HTML."""
        if self.json_ld:
            desc = self.json_ld.get("description")
            if desc:
                return desc.strip()
        # Try common description divs
        for selector in [".product-description", ".description", "[data-product-description]"]:
            el = self.soup.select_one(selector)
            if el:
                return el.get_text(strip=True)
        # Meta description
        meta = self.soup.find("meta", attrs={"name": "description"})
        if meta:
            return meta.get("content", "").strip()
        return None

    def extract_image_urls(self) -> tuple[str | None, list[str]]:
        """
        Return (main_image_url, [additional_image_urls]).
        Gets all unique product images from the Shopify CDN.
        """
        all_image_urls: list[str] = []
        seen: set[str] = set()

        # 1. JSON-LD image
        if self.json_ld:
            img = self.json_ld.get("image")
            if img:
                if isinstance(img, str) and img not in seen:
                    all_image_urls.append(img)
                    seen.add(img)
                elif isinstance(img, list):
                    for i in img:
                        if isinstance(i, str) and i not in seen:
                            all_image_urls.append(i)
                            seen.add(i)

        # 2. OG image meta
        for prop in ["og:image", "og:image:secure_url"]:
            meta = self.soup.find("meta", property=prop)
            if meta:
                url = meta.get("content", "")
                if url and url not in seen:
                    all_image_urls.append(url)
                    seen.add(url)

        # 3. Twitter image
        meta = self.soup.find("meta", attrs={"name": "twitter:image"})
        if meta:
            url = meta.get("content", "")
            if url and url not in seen:
                all_image_urls.append(url)
                seen.add(url)

        # 4. All <img> tags with Shopify CDN product images
        for img in self.soup.find_all("img"):
            for attr in ["src", "data-src", "data-original"]:
                val = img.get(attr, "")
                if "cdn.shopify.com" in val and "/products/" in val:
                    # Normalize URL — remove query params for dedup
                    clean = val.split("?")[0].split("&")[0]
                    # Use the full-size version by removing width params
                    clean = clean.replace("_{width}x", "")
                    if clean not in seen:
                        all_image_urls.append(clean)
                        seen.add(clean)
                    break

        # Remove duplicates while preserving order
        main = all_image_urls[0] if all_image_urls else None
        additional = all_image_urls[1:] if len(all_image_urls) > 1 else []
        return main, additional

    def extract_prices(self) -> tuple[str | None, str | None]:
        """
        Return (original_price_str, sale_price_str) in 'X.XXUSD' format.
        Detects sale prices from compare_at_price in Shopify data.
        """
        # First, try to get compare_at_price from Shopify Analytics meta
        # (most reliable source for sale detection)
        compare_at = None
        regular_price = None

        for script in self.soup.find_all("script"):
            if script.string and "ShopifyAnalytics" in script.string:
                # Look for product variants with compare_at_price
                match_variants = re.search(r'variants"\s*:\s*(\[[^\]]+\])', script.string)
                if match_variants:
                    try:
                        variants = json.loads(match_variants.group(1))
                        if isinstance(variants, list) and len(variants) > 0:
                            prices = []
                            compare_prices = []
                            for v in variants:
                                if isinstance(v, dict):
                                    p = v.get("price")
                                    if p is not None:
                                        # Shopify Analytics stores prices in cents
                                        prices.append(float(p) / 100.0)
                                    cp = v.get("compare_at_price")
                                    if cp is not None and float(cp) > 0:
                                        # Shopify Analytics stores prices in cents
                                        compare_prices.append(float(cp) / 100.0)
                            if prices:
                                regular_price = min(prices)
                            if compare_prices:
                                compare_at = max(compare_prices)
                    except (json.JSONDecodeError, AttributeError, ValueError):
                        pass

        # Fallback: JSON-LD offers
        if regular_price is None and self.json_ld:
            offers = self.json_ld.get("offers", {})
            if isinstance(offers, dict):
                p = offers.get("price")
                if p is not None:
                    regular_price = float(p)
            elif isinstance(offers, list):
                prices = []
                for offer in offers:
                    if isinstance(offer, dict):
                        p = offer.get("price")
                        if p is not None:
                            prices.append(float(p))
                if prices:
                    regular_price = min(prices)

        # Fallback: meta tag
        if regular_price is None:
            price_meta = self.soup.find("meta", property="product:price:amount")
            if price_meta:
                try:
                    regular_price = float(price_meta.get("content", "0"))
                except ValueError:
                    pass

        if regular_price is None:
            return None, None

        # If compare_at_price > price, it's a sale: price is sale, compare_at is original
        if compare_at is not None and compare_at > regular_price:
            original_str = normalize_price(str(compare_at))
            sale_str = normalize_price(str(regular_price))
            return original_str, sale_str

        # No sale
        return normalize_price(str(regular_price)), None

    def extract_variants(self) -> list[dict]:
        """Extract variant data (sizes, prices) from JSON-LD or analytics."""
        variants = []

        # Try JSON-LD offers
        if self.json_ld:
            offers = self.json_ld.get("offers", [])
            if isinstance(offers, list):
                for offer in offers:
                    if isinstance(offer, dict):
                        variants.append({
                            "name": offer.get("name", ""),
                            "sku": offer.get("sku", ""),
                            "price": offer.get("price"),
                            "currency": offer.get("priceCurrency", "USD"),
                            "availability": offer.get("availability", ""),
                        })
            elif isinstance(offers, dict):
                variants.append({
                    "name": "Default",
                    "sku": offers.get("sku", ""),
                    "price": offers.get("price"),
                    "currency": offers.get("priceCurrency", "USD"),
                    "availability": offers.get("availability", ""),
                })

        return variants

    def extract_sizes(self) -> str | None:
        """Extract available sizes as comma-separated string."""
        variants = self.extract_variants()
        sizes = []
        for v in variants:
            name = v.get("name", "")
            # Shopify often uses size names like "S", "M", "L", "XL"
            if name and name != "Default":
                sizes.append(name)

        # Also try to find size selector in HTML
        if not sizes:
            select = self.soup.find("select", attrs={"name": "id"})
            if select:
                for option in select.find_all("option"):
                    text = option.get_text(strip=True)
                    if text and text not in sizes:
                        sizes.append(text)

        if sizes:
            return ", ".join(sizes)
        return None

    def extract_tags(self) -> list[str] | None:
        """Extract product tags if available."""
        tags = []
        # Try to find tags in meta or script data
        for script in self.soup.find_all("script"):
            if script.string and "tags" in script.string:
                found = re.findall(r'"tags":\s*\[([^\]]+)\]', script.string)
                for match in found:
                    tag_list = [t.strip().strip('"').strip("'") for t in match.split(",")]
                    tags.extend(t for t in tag_list if t and t not in tags)                # Also check Shopify Analytics meta for collections/tags
        for script in self.soup.find_all("script"):
            if script.string and "ShopifyAnalytics" in script.string:
                match_meta = re.search(r'meta\s*:\s*(\{.*\})', script.string)
                if match_meta:
                    try:
                        meta = json.loads(match_meta.group(1))
                        product_data = meta.get("product", {})
                        if isinstance(product_data, dict):
                            product_tags = product_data.get("tags", [])
                            if isinstance(product_tags, list):
                                tags.extend(t for t in product_tags if t not in tags)
                    except (json.JSONDecodeError, AttributeError):
                        pass

        return tags if tags else None

    def extract_category(self) -> str | None:
        """Extract product category. Try product type, tags, or breadcrumbs."""
        # 1. Check Shopify Analytics meta for product type
        for script in self.soup.find_all("script"):
            if script.string and "ShopifyAnalytics" in script.string:
                match = re.search(r'product_type["\']\s*:\s*["\']([^"\']+)', script.string)
                if match:
                    category = match.group(1).strip()
                    if category:
                        # Check for compound categories like "Sweaters & Hoodies"
                        if " & " in category:
                            return category.replace(" & ", ", ")
                        return category

        # 2. Check tags for category-like tags
        tags = self.extract_tags()
        if tags:
            # Common Shopify category-like tags (substring matching for compound tags)
            category_keywords = [
                "tops", "bottoms", "outerwear", "headwear", "accessories",
                "tees", "sweats", "hoodies", "crewnecks", "sweatpants",
                "jackets", "shirts", "pants", "shorts", "hats", "bags",
            ]
            found = []
            for t in tags:
                tl = t.lower()
                for kw in category_keywords:
                    if kw in tl:
                        found.append(kw.title())
                        break
            if found:
                return ", ".join(dict.fromkeys(found))

        # 3. Check for breadcrumbs
        breadcrumbs = self.soup.select('[class*="breadcrumb"] a, [class*="breadcrumb"] span')
        if breadcrumbs:
            crumbs = [b.get_text(strip=True) for b in breadcrumbs if b.get_text(strip=True)]
            if len(crumbs) >= 2:
                return crumbs[-2]

        return None

    def extract_metadata(self) -> str:
        """Build a comprehensive metadata JSON string."""
        variants = self.extract_variants()
        sizes = self.extract_sizes()
        tags = self.extract_tags()
        title = self.extract_title()
        description = self.extract_description()
        main_img, additional_imgs = self.extract_image_urls()
        price, sale = self.extract_prices()
        category = self.extract_category()

        meta = {
            "name": title,
            "description": description or "",
            "sizes": sizes.split(", ") if sizes else [],
            "price": price or "",
            "sale_price": sale or "",
            "currency": "USD",
            "variants": variants,
            "tags": tags or [],
            "category": category or "",
            "product_url": self.url,
            "image_url": main_img or "",
            "additional_images": additional_imgs,
            "source": SOURCE_NAME,
            "brand": BRAND_NAME,
        }
        return json.dumps(meta, indent=2)

    def extract_gender(self) -> str | None:
        """Determine gender from product context. Default unisex for streetwear."""
        return "unisex"

    def to_supabase_record(self) -> dict[str, Any] | None:
        """Convert scraped product data to a Supabase-ready record."""
        title = self.extract_title()
        description = self.extract_description()
        main_img, additional_imgs = self.extract_image_urls()
        price, sale = self.extract_prices()
        category = self.extract_category()
        sizes = self.extract_sizes()
        tags = self.extract_tags()
        metadata = self.extract_metadata()

        if not main_img and not title:
            log.warning(f"No image or title found for {self.url}")
            return None

        product_id = make_id(self.url)

        # Sort for deterministic comparison across runs
        additional_images_str = (
            " , ".join(sorted(additional_imgs)) if additional_imgs else None
        )

        record = {
            "id": product_id,
            "source": SOURCE_NAME,
            "product_url": self.url,
            "affiliate_url": None,
            "image_url": main_img or "",
            "brand": BRAND_NAME,
            "title": title,
            "description": description,
            "category": category,
            "gender": self.extract_gender(),
            "search_tsv": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata,
            "size": sizes,
            "second_hand": False,
            "image_embedding": None,  # Will be filled later
            "country": None,
            "compressed_image_url": None,
            "tags": tags,
            "search_vector": None,
            "title_tsv": None,
            "brand_tsv": None,
            "description_tsv": None,
            "other": None,
            "price": price,
            "sale": sale,
            "additional_images": additional_images_str,
            "info_embedding": None,  # Will be filled later
        }
        return record


# ---------------------------------------------------------------------------
# Embedding Generator
# ---------------------------------------------------------------------------

class EmbeddingGenerator:
    """Generate image and text embeddings using google/siglip-base-patch16-384."""

    def __init__(self, device: str | None = None) -> None:
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        log.info(f"Loading SigLIP model on {self.device}...")
        self.processor = SiglipProcessor.from_pretrained(MODEL_NAME)
        self.model = SiglipModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()
        log.info("SigLIP model loaded successfully.")

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> list[float]:
        """Generate a 768-dim image embedding."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        # get_image_features returns BaseModelOutputWithPooling — extract pooler_output
        if hasattr(outputs, "pooler_output"):
            emb_tensor = outputs.pooler_output
        else:
            emb_tensor = outputs[0]  # fallback: last_hidden_state
        embedding = emb_tensor.cpu().numpy().flatten()
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()

    @torch.no_grad()
    def embed_text(self, text: str) -> list[float]:
        """Generate a 768-dim text embedding from product info.
        SigLIP text model has max_position_embeddings=64, so we keep text short.
        """
        inputs = self.processor(
            text=text,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model.get_text_features(**inputs)
        # get_text_features returns BaseModelOutputWithPooling — extract pooler_output
        if hasattr(outputs, "pooler_output"):
            emb_tensor = outputs.pooler_output
        else:
            emb_tensor = outputs[0]  # fallback: last_hidden_state
        embedding = emb_tensor.cpu().numpy().flatten()
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()

    def make_info_text(self, record: dict) -> str:
        """Build a comprehensive text string for info embedding."""
        parts = []
        if record.get("title"):
            parts.append(f"Title: {record['title']}")
        if record.get("brand"):
            parts.append(f"Brand: {record['brand']}")
        if record.get("description"):
            parts.append(f"Description: {record['description']}")
        if record.get("category"):
            parts.append(f"Category: {record['category']}")
        if record.get("price"):
            parts.append(f"Price: {record['price']}")
        if record.get("sale"):
            parts.append(f"Sale: {record['sale']}")
        if record.get("size"):
            parts.append(f"Sizes: {record['size']}")
        if record.get("gender"):
            parts.append(f"Gender: {record['gender']}")
        if record.get("tags"):
            parts.append(f"Tags: {', '.join(record['tags'])}")
        if record.get("metadata"):
            try:
                meta = json.loads(record["metadata"])
                if meta.get("variants"):
                    vnames = [v.get("name", "") for v in meta["variants"]]
                    parts.append(f"Variants: {', '.join(vnames)}")
                if meta.get("additional_images"):
                    parts.append(f"Additional images count: {len(meta['additional_images'])}")
            except (json.JSONDecodeError, TypeError):
                parts.append(f"Metadata: {record['metadata'][:500]}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Supabase Client
# ---------------------------------------------------------------------------

class SupabaseUploader:
    """
    Upload product records to Supabase with batch upsert support,
    stale product cleanup, and error logging.
    """

    def __init__(self) -> None:
        log.info("Connecting to Supabase...")
        self.client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        # Verify connection
        try:
            self.client.table("products").select("id").limit(1).execute()
            log.info("Supabase connection OK.")
        except Exception as e:
            log.warning(f"Could not verify Supabase connection: {e}")

    def fetch_existing_products(self, source: str) -> list[dict]:
        """Fetch ALL existing products for a given source."""
        try:
            result = self.client.table("products").select("*").eq("source", source).execute()
            data = result.data or []
            log.info(f"Fetched {len(data)} existing products for source '{source}'.")
            return data
        except Exception as e:
            log.error(f"Failed to fetch existing products: {e}")
            return []

    def batch_upsert(self, records: list[dict], batch_size: int = 50) -> tuple[int, int]:
        """
        Upsert records in batches. Returns (success_count, fail_count).
        Retries failed batches up to MAX_RETRIES times before logging
        the failed products to a local file.
        """
        if not records:
            return 0, 0

        success_count = 0
        fail_count = 0
        total_batches = (len(records) + batch_size - 1) // batch_size

        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            batch_num = i // batch_size + 1
            batch_ok = False

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    self.client.table("products").upsert(
                        batch, on_conflict="source,product_url"
                    ).execute()
                    success_count += len(batch)
                    if total_batches > 1:
                        log.info(f"  Batch {batch_num}/{total_batches} upserted ({len(batch)} products).")
                    batch_ok = True
                    break
                except Exception as e:
                    log.warning(
                        f"  Batch {batch_num}/{total_batches} "
                        f"attempt {attempt}/{MAX_RETRIES} failed: {e}"
                    )
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY * attempt)

            if not batch_ok:
                fail_count += len(batch)
                self._log_failed_products(batch, "All retries exhausted")
                log.error(
                    f"  Batch {batch_num}/{total_batches} FAILED after "
                    f"{MAX_RETRIES} attempts. Products logged to {FAILED_LOG_FILE}."
                )

        return success_count, fail_count

    def _log_failed_products(self, records: list[dict], error: str) -> None:
        """Append failed product records to a JSONL log file."""
        timestamp = datetime.now(timezone.utc).isoformat()
        lines = []
        for rec in records:
            entry = {
                "timestamp": timestamp,
                "error": error,
                "source": rec.get("source"),
                "product_url": rec.get("product_url"),
                "id": rec.get("id"),
                "title": rec.get("title"),
            }
            lines.append(json.dumps(entry))
        try:
            with open(FAILED_LOG_FILE, "a") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as log_err:
            log.error(f"  Failed to write error log: {log_err}")

    def cleanup_stale_products(self, source: str, seen_urls: set[str]) -> int:
        """
        Identify and handle stale products for this source.
        Products not seen this run get their consecutive_misses incremented.
        Products with consecutive_misses >= STALE_THRESHOLD get deleted.
        Returns the number of deleted products.
        """
        try:
            result = self.client.table("products").select(
                "id, product_url, consecutive_misses"
            ).eq("source", source).execute()
            existing = result.data or []
        except Exception as e:
            log.error(f"Failed to fetch products for stale cleanup: {e}")
            return 0

        if not existing:
            return 0

        miss_updates: list[dict] = []
        delete_ids: list[str] = []

        for product in existing:
            url = product.get("product_url", "")
            if url not in seen_urls:
                misses = (product.get("consecutive_misses") or 0) + 1
                pid = product["id"]
                if misses >= STALE_THRESHOLD:
                    delete_ids.append(pid)
                else:
                    miss_updates.append({
                        "id": pid,
                        "consecutive_misses": misses,
                    })

        # Update miss counters for products not yet stale
        for update in miss_updates:
            try:
                self.client.table("products").update({
                    "consecutive_misses": update["consecutive_misses"],
                }).eq("id", update["id"]).execute()
            except Exception as e:
                log.warning(f"  Failed to update misses for {update['id']}: {e}")

        if miss_updates:
            log.info(f"  Incremented misses for {len(miss_updates)} unseen products.")

        # Delete stale products (deleted in batches of 50)
        deleted_count = 0
        for i in range(0, len(delete_ids), 50):
            batch = delete_ids[i:i + 50]
            try:
                self.client.table("products").delete().in_("id", batch).execute()
                deleted_count += len(batch)
            except Exception as e:
                log.error(f"  Failed to delete batch of {len(batch)} products: {e}")

        if delete_ids:
            log.info(f"  Deleted {deleted_count}/{len(delete_ids)} stale products.")

        return deleted_count


# ---------------------------------------------------------------------------
# Main Scraper
# ---------------------------------------------------------------------------

class AkimboScraper:
    """Orchestrate the full scraping pipeline."""

    def __init__(
        self,
        limit: int | None = None,
        resume: bool = False,
    ) -> None:
        self.limit = limit
        self.resume = resume
        self.checkpoint = load_checkpoint() if resume else {"done": [], "failed": []}
        self.stats = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "total_products": 0,
            "scraped": 0,
            "embedded": 0,
            "uploaded": 0,
            "failed": len(self.checkpoint.get("failed", [])),
            "errors": [],
        }

        self.http_client: httpx.AsyncClient | None = None
        self.embedder: EmbeddingGenerator | None = None
        self.uploader: SupabaseUploader | None = None
        self._sem_fetch: asyncio.Semaphore | None = None
        self._sem_embed: asyncio.Semaphore | None = None

    async def __aenter__(self):
        self.http_client = httpx.AsyncClient(
            headers=HEADERS,
            follow_redirects=True,
            timeout=30.0,
        )
        self.embedder = EmbeddingGenerator()
        self.uploader = SupabaseUploader()
        self._sem_fetch = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)
        self._sem_embed = asyncio.Semaphore(MAX_CONCURRENT_EMBEDDINGS)
        return self

    async def __aexit__(self, *args):
        if self.http_client:
            await self.http_client.aclose()

    # ---- Fetching ----

    async def fetch(self, url: str) -> str | None:
        """Fetch a URL with retry logic."""
        for attempt in range(MAX_RETRIES):
            try:
                async with self._sem_fetch:
                    resp = await self.http_client.get(url)
                    resp.raise_for_status()
                    return resp.text
            except Exception as e:
                log.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {url}: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
        return None

    async def fetch_image(self, url: str) -> Image.Image | None:
        """Fetch and open a product image."""
        for attempt in range(MAX_RETRIES):
            try:
                async with self._sem_fetch:
                    resp = await self.http_client.get(url)
                    resp.raise_for_status()
                    return Image.open(BytesIO(resp.content)).convert("RGB")
            except Exception as e:
                log.warning(f"Image fetch attempt {attempt + 1} failed for {url}: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
        return None

    # ---- Collection scraping ----

    async def get_product_urls(self) -> list[str]:
        """Scrape collection page for all product URLs."""
        html = await self.fetch(COLLECTION_URL)
        if not html:
            log.error("Failed to fetch collection page!")
            return []

        soup = BeautifulSoup(html, "lxml")
        urls: list[str] = []
        seen: set[str] = set()

        # Extract from <a> tags with /products/
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/products/" in href and "/products/" not in href.split("/products/")[0]:
                # Normalize to full URL
                if href.startswith("/"):
                    full = f"{BASE_URL}{href}"
                elif href.startswith("http"):
                    full = href
                else:
                    full = f"{BASE_URL}/{href}"
                # Remove any query params or anchors
                full = full.split("?")[0].split("#")[0]
                if full not in seen and "physical-gift-card" not in full.lower():
                    seen.add(full)
                    urls.append(full)

        # Also try to find product handles in embedded Shopify data
        for script in soup.find_all("script"):
            if script.string and "ShopifyAnalytics" in script.string:
                match = re.search(r'products\s*:\s*(\[[^\]]+\])', script.string)
                if match:
                    try:
                        products = json.loads(match.group(1))
                        for p in products:
                            if isinstance(p, dict) and "handle" in p:
                                full = f"{BASE_URL}/products/{p['handle']}"
                                if full not in seen:
                                    seen.add(full)
                                    urls.append(full)
                    except (json.JSONDecodeError, AttributeError):
                        pass

        # Deduplicate while preserving order
        seen_dedup: set[str] = set()
        unique_urls: list[str] = []
        for u in urls:
            if u not in seen_dedup:
                seen_dedup.add(u)
                unique_urls.append(u)

        log.info(f"Found {len(unique_urls)} unique product URLs on collection page.")
        return unique_urls

    # ---- Processing pipeline ----

    async def process_product(self, url: str, existing_by_url: dict) -> dict | None:
        """
        Scrape a single product page, compare against existing DB record,
        generate embeddings only if needed, and return structured result.
        Returns None on failure, or a dict with status ("new"/"updated"/"skipped") and record.
        """
        pid = make_id(url)

        log.info(f"--- Processing: {url.split('/')[-1]} ---")

        # 1. Fetch product page
        html = await self.fetch(url)
        if not html:
            self.checkpoint.setdefault("failed", []).append(pid)
            save_checkpoint(self.checkpoint)
            return None

        # 2. Parse product data
        scraper = ProductScraper(url, html)
        record = scraper.to_supabase_record()
        if not record:
            log.warning(f"Could not extract data from {url}")
            self.checkpoint.setdefault("failed", []).append(pid)
            save_checkpoint(self.checkpoint)
            return None

        # 3. Check if product already exists and whether anything changed
        existing = existing_by_url.get(url)
        if existing:
            # Preserve the original created_at timestamp
            record["created_at"] = existing.get("created_at", record["created_at"])

            if not product_changed(record, existing):
                log.info(f"  Product unchanged — skipping entirely.")
                return {"status": "skipped", "record": None}
            else:
                log.info(f"  Product changed — will update.")

        # 4. Set tracking fields
        record["last_seen"] = datetime.now(timezone.utc).isoformat()
        record["consecutive_misses"] = 0

        # 5. Generate embeddings only if new or the image URL has changed
        needs_embeddings = existing is None or image_url_changed(record, existing)

        if needs_embeddings:
            # Image embedding
            image_url = record.get("image_url")
            if image_url:
                log.info("  Generating image embedding...")
                async with self._sem_embed:
                    img = await self.fetch_image(image_url)
                    if img:
                        try:
                            emb = self.embedder.embed_image(img)
                            record["image_embedding"] = emb
                            log.info(f"  Image embedding done (dim={len(emb)})")
                        except Exception as e:
                            log.error(f"  Image embedding failed: {e}")
                            traceback.print_exc()
                await asyncio.sleep(STAGGER_DELAY)

            # Text info embedding
            info_text = self.embedder.make_info_text(record)
            log.info("  Generating text info embedding...")
            try:
                text_emb = self.embedder.embed_text(info_text)
                record["info_embedding"] = text_emb
                log.info(f"  Text embedding done (dim={len(text_emb)})")
            except Exception as e:
                log.error(f"  Text embedding failed: {e}")
                traceback.print_exc()
            await asyncio.sleep(STAGGER_DELAY)
        else:
            log.info("  Image URL unchanged — keeping existing embeddings.")
            record["image_embedding"] = existing.get("image_embedding")
            record["info_embedding"] = existing.get("info_embedding")

        status = "new" if existing is None else "updated"
        return {"status": status, "record": record}

    async def run(self) -> None:
        """Run the full scraping pipeline with batch upsert and stale cleanup."""
        log.info("=" * 60)
        log.info("AKIMBO CLUB SCRAPER")
        log.info("=" * 60)

        # 1. Fetch existing products for comparison and upsert targeting
        existing_products = self.uploader.fetch_existing_products(SOURCE_NAME)
        existing_by_url: dict[str, dict] = {p["product_url"]: p for p in existing_products}
        log.info(f"Loaded {len(existing_products)} existing products from DB.")

        # 2. Get product URLs from collection page
        all_urls = await self.get_product_urls()
        seen_urls: set[str] = set()
        if self.limit:
            all_urls = all_urls[:self.limit]

        self.stats["total_products"] = len(all_urls)
        log.info(f"Will process {len(all_urls)} products")

        # 3. Process products — collect into batches
        upsert_batch: list[dict] = []
        stats_new = 0
        stats_updated = 0
        stats_skipped = 0

        start_time = time.time()

        for i, url in enumerate(all_urls, 1):
            pid = make_id(url)
            if pid in self.checkpoint.get("done", []):
                log.info(f"[{i}/{len(all_urls)}] Already done, skipping: {url.split('/')[-1]}")
                stats_skipped += 1
                seen_urls.add(url)
                continue

            log.info(f"[{i}/{len(all_urls)}] Processing...")
            result = await self.process_product(url, existing_by_url)

            if result is None:
                # Fetch or parse failure
                continue

            seen_urls.add(url)

            if result["status"] == "skipped":
                stats_skipped += 1
                # Update checkpoint — mark as done even if unchanged
                self.checkpoint.setdefault("done", []).append(pid)
                save_checkpoint(self.checkpoint)
            else:
                upsert_batch.append(result["record"])
                if result["status"] == "new":
                    stats_new += 1
                else:
                    stats_updated += 1

            # Flush batch when it reaches BATCH_SIZE
            if len(upsert_batch) >= BATCH_SIZE:
                succ, _ = self.uploader.batch_upsert(upsert_batch)
                for rec in upsert_batch[:succ]:
                    rec_pid = rec["id"]
                    if rec_pid not in self.checkpoint.get("done", []):
                        self.checkpoint.setdefault("done", []).append(rec_pid)
                save_checkpoint(self.checkpoint)
                upsert_batch = []

            # Periodic progress update
            elapsed = time.time() - start_time
            done_count = len(self.checkpoint.get("done", []))
            rate = done_count / elapsed if elapsed > 0 else 0
            eta = (len(all_urls) - done_count) / rate if rate > 0 else 0
            self.stats["elapsed"] = elapsed
            self.stats["rate_per_sec"] = round(rate, 2)
            self.stats["eta_seconds"] = round(eta)
            save_progress(self.stats)

        # 4. Flush remaining batch
        if upsert_batch:
            succ, _ = self.uploader.batch_upsert(upsert_batch)
            for rec in upsert_batch[:succ]:
                rec_pid = rec["id"]
                if rec_pid not in self.checkpoint.get("done", []):
                    self.checkpoint.setdefault("done", []).append(rec_pid)
            save_checkpoint(self.checkpoint)

        # 5. Clean up stale products
        log.info("=" * 60)
        log.info("STALE PRODUCT CLEANUP")
        log.info("=" * 60)
        stats_deleted = self.uploader.cleanup_stale_products(SOURCE_NAME, seen_urls)

        # 6. Print run summary
        elapsed = time.time() - start_time
        log.info("=" * 60)
        log.info("RUN SUMMARY")
        log.info("=" * 60)
        log.info(f"  New products added:     {stats_new}")
        log.info(f"  Products updated:      {stats_updated}")
        log.info(f"  Products unchanged:    {stats_skipped}")
        log.info(f"  Stale products deleted: {stats_deleted}")
        log.info(f"  Elapsed:               {elapsed:.1f}s")
        log.info("=" * 60)

        save_progress({
            **self.stats,
            "new": stats_new,
            "updated": stats_updated,
            "skipped": stats_skipped,
            "deleted": stats_deleted,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Akimbo Club Scraper")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of products to process")
    args = parser.parse_args()

    async def run():
        async with AkimboScraper(limit=args.limit, resume=args.resume) as scraper:
            await scraper.run()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("Interrupted by user. Progress saved in checkpoint.")
        sys.exit(1)


if __name__ == "__main__":
    main()
