"""
Migrate old redundant schema to new optimized schema
Process in batches to avoid memory issues
"""
import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.utils.mongo import db


def transform_document(old_doc: dict) -> dict:
    """Transform old schema to new optimized schema"""
    
    # Pick first non-null experience value
    exp_years = float(
        old_doc.get("experience") or
        old_doc.get("total_experience_years") or
        old_doc.get("yoe") or
        old_doc.get("years_of_experience") or
        0.0
    )
    
    # Pick role fields
    role_current = (
        old_doc.get("current_title") or
        old_doc.get("title") or
        old_doc.get("job_title") or
        None
    )
    
    role_predicted = (
        old_doc.get("predicted_role") or
        old_doc.get("category") or
        role_current or
        ""
    )
    
    # Pick confidence (already 0-1)
    confidence = float(
        old_doc.get("ml_confidence") or
        old_doc.get("role_confidence") or
        0.0
    )
    
    # Normalize confidence if it's in 0-100 range
    if confidence > 1.0:
        confidence = confidence / 100.0
    
    # Experience display
    if 0 < exp_years < 1:
        exp_display = "< 1 year"
    elif exp_years >= 1:
        yrs_int = int(round(exp_years))
        exp_display = f"{yrs_int} year" if yrs_int == 1 else f"{yrs_int} years"
    else:
        exp_display = "Not specified"
    
    # Location parsing
    location_full = old_doc.get("location", "N/A")
    city = old_doc.get("city")
    country = old_doc.get("country")
    
    # Parse city/country if not present
    if not city and location_full and location_full != "N/A":
        parts = [p.strip() for p in location_full.split(",") if p.strip()]
        if len(parts) >= 2:
            city = parts[0]
            country = parts[-1]
    
    # Skills normalization
    skills = old_doc.get("skills", [])
    if not isinstance(skills, list):
        skills = []
    skills_norm = old_doc.get("skills_norm", [s.lower().strip() for s in skills])
    
    # Projects normalization
    projects_list = []
    if isinstance(old_doc.get("resume"), dict):
        projects_list = old_doc["resume"].get("projects", [])
    elif isinstance(old_doc.get("projects"), list):
        projects_list = old_doc["projects"]
    
    projects_norm = old_doc.get("projects_norm", [])
    
    # Education
    degrees = old_doc.get("degrees_detected", [])
    institutions = old_doc.get("schools_detected", [])
    
    # Work history
    work_history = []
    if isinstance(old_doc.get("resume"), dict):
        work_history = old_doc["resume"].get("workHistory", [])
    
    # Search text
    search_text = (
        old_doc.get("search_blob") or 
        old_doc.get("index_blob") or 
        ""
    )[:4096]
    
    full_text = (old_doc.get("raw_text") or "")[:10000]
    
    return {
        "_id": old_doc["_id"],
        "ownerUserId": old_doc.get("ownerUserId", ""),
        "filename": old_doc.get("filename", ""),
        "contentHash": old_doc.get("content_hash", ""),
        "uploadedAt": old_doc.get("created_at") or datetime.utcnow(),
        "updatedAt": datetime.utcnow(),
        
        "personal": {
            "name": old_doc.get("name", "No Name"),
            "email": old_doc.get("email"),
            "phone": old_doc.get("phone"),
            "location": {
                "full": location_full,
                "city": city,
                "country": country,
                "normalized": (location_full or "").lower().strip()
            }
        },
        
        "role": {
            "current": role_current,
            "predicted": role_predicted,
            "confidence": round(confidence, 3),
            "normalized": old_doc.get("role_norm", (role_predicted or "").lower().strip())
        },
        
        "experience": {
            "years": float(exp_years),
            "display": exp_display
        },
        
        "skills": {
            "list": skills,
            "normalized": skills_norm,
            "count": len(skills)
        },
        
        "education": {
            "degrees": degrees,
            "institutions": institutions,
            "normalized": degrees
        },
        
        "projects": {
            "list": projects_list,
            "normalized": projects_norm
        },
        
        "workHistory": work_history,
        
        "ml": {
            "category": old_doc.get("category"),
            "categoryConfidence": float(old_doc.get("category_confidence", 0.0)),
            "relatedRoles": old_doc.get("related_roles", []),
            "embedding": old_doc.get("index_embedding"),
            "embeddingModel": "all-MiniLM-L6-v2",
            "lastEmbeddingUpdate": old_doc.get("embedding_updated_at")
        },
        
        "search": {
            "text": search_text,
            "fullText": full_text
        },
        
        "matching": {
            "promptScore": old_doc.get("final_score", 0),
            "semanticScore": float(old_doc.get("semantic_score", 0.0)),
            "lastPrompt": old_doc.get("last_prompt"),
            "lastMatchedAt": old_doc.get("last_matched_at")
        }
    }


async def migrate_all(batch_size=1000, dry_run=False):
    """Migrate all documents from old to new schema"""
    
    print("=" * 60)
    print("SCHEMA MIGRATION: OLD → NEW")
    print("=" * 60)
    
    if dry_run:
        print("\n⚠️  DRY RUN MODE - No data will be written\n")
    else:
        print("\n⚠️  LIVE MODE - Data will be migrated\n")
    
    total = await db.parsed_resumes.count_documents({})
    print(f"📊 Total documents to migrate: {total:,}")
    
    if total == 0:
        print("\n❌ No documents found in 'parsed_resumes' collection")
        return
    
    # Check if destination collection exists
    existing = await db.resumes.count_documents({})
    if existing > 0:
        print(f"\n⚠️  WARNING: 'resumes' collection already has {existing:,} documents")
        response = input("Continue and add more documents? (yes/no): ")
        if response.lower() != "yes":
            print("Migration cancelled.")
            return
    
    print(f"\n🚀 Starting migration in batches of {batch_size}...")
    print("-" * 60)
    
    migrated = 0
    errors = 0
    batch = []
    
    async for doc in db.parsed_resumes.find({}):
        try:
            new_doc = transform_document(doc)
            batch.append(new_doc)
            
            if len(batch) >= batch_size:
                if not dry_run:
                    await db.resumes.insert_many(batch)
                migrated += len(batch)
                progress = (migrated / total) * 100
                print(f"  ✓ Migrated {migrated:,}/{total:,} documents ({progress:.1f}%)")
                batch = []
        
        except Exception as e:
            errors += 1
            doc_id = doc.get('_id', 'unknown')
            print(f"  ✗ Error migrating doc {doc_id}: {e}")
            continue
    
    # Insert remaining documents
    if batch:
        if not dry_run:
            await db.resumes.insert_many(batch)
        migrated += len(batch)
        progress = (migrated / total) * 100
        print(f"  ✓ Migrated {migrated:,}/{total:,} documents ({progress:.1f}%)")
    
    print("-" * 60)
    
    if dry_run:
        print(f"\n✅ DRY RUN COMPLETE")
        print(f"  • Would migrate: {migrated:,} documents")
        print(f"  • Would skip: {errors} documents (errors)")
    else:
        print(f"\n✅ MIGRATION COMPLETE")
        print(f"  • Migrated: {migrated:,} documents")
        print(f"  • Errors: {errors} documents")
        print(f"  • Success rate: {(migrated/(migrated+errors)*100):.1f}%")
    
    # Verify migration
    if not dry_run:
        new_count = await db.resumes.count_documents({})
        print(f"\n📊 Verification:")
        print(f"  • Old collection: {total:,} documents")
        print(f"  • New collection: {new_count:,} documents")
        
        if new_count >= total:
            print("\n✅ Migration verified successfully!")
        else:
            print(f"\n⚠️  WARNING: Document count mismatch!")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Run validation: python -m app.scripts.validate_migration")
    print("2. Create indexes: python -m app.scripts.optimize_indexes")
    print("3. Test queries to verify performance improvement")
    print("4. Keep old collection for 2 weeks as backup")
    print("5. After verification, drop old collection:")
    print("   db.parsed_resumes.drop()")


async def sample_comparison(limit=5):
    """Show side-by-side comparison of old vs new schema"""
    print("\n" + "=" * 60)
    print("SAMPLE DOCUMENT COMPARISON")
    print("=" * 60)
    
    print(f"\nShowing first {limit} documents:")
    
    count = 0
    async for old_doc in db.parsed_resumes.find({}).limit(limit):
        count += 1
        new_doc = transform_document(old_doc)
        
        print(f"\n--- Document {count} ---")
        print(f"Old size: {len(str(old_doc))} bytes")
        print(f"New size: {len(str(new_doc))} bytes")
        print(f"Reduction: {((1 - len(str(new_doc))/len(str(old_doc))) * 100):.1f}%")
        
        print(f"\nOld schema (excerpt):")
        print(f"  name: {old_doc.get('name')}")
        print(f"  predicted_role: {old_doc.get('predicted_role')}")
        print(f"  experience: {old_doc.get('experience')}")
        print(f"  total_experience_years: {old_doc.get('total_experience_years')}")
        print(f"  yoe: {old_doc.get('yoe')}")
        
        print(f"\nNew schema (excerpt):")
        print(f"  personal.name: {new_doc['personal']['name']}")
        print(f"  role.predicted: {new_doc['role']['predicted']}")
        print(f"  experience.years: {new_doc['experience']['years']}")
        print(f"  experience.display: {new_doc['experience']['display']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate resume schema from old to new")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing data")
    parser.add_argument("--sample", action="store_true", help="Show sample comparison only")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for migration")
    
    args = parser.parse_args()
    
    if args.sample:
        asyncio.run(sample_comparison())
    else:
        asyncio.run(migrate_all(batch_size=args.batch_size, dry_run=args.dry_run))



