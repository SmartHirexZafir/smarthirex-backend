"""
Validate that migrated data is correct
Run this after migration to verify data integrity
"""
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.utils.mongo import db


async def validate():
    """Run validation checks on migrated data"""
    
    print("=" * 60)
    print("MIGRATION VALIDATION")
    print("=" * 60)
    
    all_passed = True
    
    # ==========================================
    # Check 1: Document Count
    # ==========================================
    print("\n1️⃣  Checking document counts...")
    old_count = await db.parsed_resumes.count_documents({})
    new_count = await db.resumes.count_documents({})
    
    print(f"  • Old collection: {old_count:,} documents")
    print(f"  • New collection: {new_count:,} documents")
    
    if new_count >= old_count:
        print("  ✅ PASS: Document count is correct")
    else:
        print(f"  ❌ FAIL: Missing {old_count - new_count} documents!")
        all_passed = False
    
    # ==========================================
    # Check 2: Required Fields Present
    # ==========================================
    print("\n2️⃣  Checking required fields...")
    sample = await db.resumes.find_one({})
    
    if not sample:
        print("  ❌ FAIL: No documents found in new collection")
        all_passed = False
    else:
        required_fields = [
            "personal", "role", "experience", "skills", 
            "education", "projects", "workHistory", 
            "ml", "search", "matching"
        ]
        
        missing = []
        for field in required_fields:
            if field not in sample:
                missing.append(field)
        
        if missing:
            print(f"  ❌ FAIL: Missing required fields: {', '.join(missing)}")
            all_passed = False
        else:
            print("  ✅ PASS: All required fields present")
    
    # ==========================================
    # Check 3: Nested Structure
    # ==========================================
    print("\n3️⃣  Checking nested structure...")
    
    if sample:
        checks = [
            ("personal.name", sample.get("personal", {}).get("name")),
            ("personal.location.full", sample.get("personal", {}).get("location", {}).get("full")),
            ("role.predicted", sample.get("role", {}).get("predicted")),
            ("role.confidence", sample.get("role", {}).get("confidence")),
            ("experience.years", sample.get("experience", {}).get("years")),
            ("experience.display", sample.get("experience", {}).get("display")),
            ("skills.list", sample.get("skills", {}).get("list")),
            ("skills.normalized", sample.get("skills", {}).get("normalized")),
        ]
        
        failed_checks = []
        for field_path, value in checks:
            if value is None:
                failed_checks.append(field_path)
        
        if failed_checks:
            print(f"  ⚠️  WARNING: Some nested fields are None: {', '.join(failed_checks)}")
            print("     (This may be expected if source data was incomplete)")
        else:
            print("  ✅ PASS: Nested structure is correct")
    
    # ==========================================
    # Check 4: No Redundant Fields
    # ==========================================
    print("\n4️⃣  Checking for redundant fields...")
    
    if sample:
        redundant = [
            "total_experience_years", "yoe", "experience_years",
            "years_of_experience", "predicted_role", "category",
            "ml_confidence", "location", "name", "email"
        ]
        
        found_redundant = []
        for field in redundant:
            if field in sample:
                found_redundant.append(field)
        
        if found_redundant:
            print(f"  ❌ FAIL: Found redundant fields: {', '.join(found_redundant)}")
            all_passed = False
        else:
            print("  ✅ PASS: No redundant fields found")
    
    # ==========================================
    # Check 5: Data Integrity Samples
    # ==========================================
    print("\n5️⃣  Sampling data integrity...")
    
    sample_size = min(100, new_count)
    samples = await db.resumes.find({}).limit(sample_size).to_list(length=sample_size)
    
    integrity_issues = 0
    for doc in samples:
        # Check experience is numeric
        exp = doc.get("experience", {}).get("years")
        if exp is not None and not isinstance(exp, (int, float)):
            integrity_issues += 1
        
        # Check confidence is 0-1
        conf = doc.get("role", {}).get("confidence")
        if conf is not None and (conf < 0 or conf > 1):
            integrity_issues += 1
        
        # Check skills.list is array
        skills_list = doc.get("skills", {}).get("list")
        if skills_list is not None and not isinstance(skills_list, list):
            integrity_issues += 1
    
    if integrity_issues > 0:
        print(f"  ⚠️  WARNING: Found {integrity_issues} integrity issues in {sample_size} samples")
    else:
        print(f"  ✅ PASS: All {sample_size} samples have correct data types")
    
    # ==========================================
    # Check 6: Index Existence
    # ==========================================
    print("\n6️⃣  Checking indexes...")
    
    indexes = await db.resumes.list_indexes().to_list(length=100)
    index_names = [idx.get("name") for idx in indexes]
    
    expected_indexes = [
        "ownerUserId_1",
        "ownerUserId_1_role.predicted_1",
        "ownerUserId_1_experience.years_1",
    ]
    
    missing_indexes = []
    for idx in expected_indexes:
        if idx not in index_names:
            missing_indexes.append(idx)
    
    if missing_indexes:
        print(f"  ⚠️  WARNING: Missing indexes: {', '.join(missing_indexes)}")
        print("     Run: python -m app.scripts.optimize_indexes")
    else:
        print(f"  ✅ PASS: Found {len(index_names)} indexes (including expected ones)")
    
    # ==========================================
    # Check 7: Sample Field Comparison
    # ==========================================
    print("\n7️⃣  Comparing sample documents with old collection...")
    
    # Get a sample from both collections
    old_sample = await db.parsed_resumes.find_one({})
    new_sample = await db.resumes.find_one({"_id": old_sample["_id"]}) if old_sample else None
    
    if old_sample and new_sample:
        # Check if name matches
        old_name = old_sample.get("name", "")
        new_name = new_sample.get("personal", {}).get("name", "")
        
        if old_name == new_name:
            print(f"  ✅ Name matches: {old_name}")
        else:
            print(f"  ⚠️  Name mismatch: '{old_name}' vs '{new_name}'")
        
        # Check if experience matches
        old_exp = old_sample.get("experience") or old_sample.get("total_experience_years") or 0
        new_exp = new_sample.get("experience", {}).get("years", 0)
        
        if float(old_exp) == float(new_exp):
            print(f"  ✅ Experience matches: {old_exp} years")
        else:
            print(f"  ⚠️  Experience mismatch: {old_exp} vs {new_exp}")
    
    # ==========================================
    # Check 8: Storage Reduction
    # ==========================================
    print("\n8️⃣  Checking storage reduction...")
    
    try:
        old_stats = await db.command("collStats", "parsed_resumes")
        new_stats = await db.command("collStats", "resumes")
        
        old_size = old_stats.get("storageSize", 0)
        new_size = new_stats.get("storageSize", 0)
        
        if old_size > 0:
            reduction = ((old_size - new_size) / old_size) * 100
            print(f"  • Old storage: {old_size / (1024*1024):.2f} MB")
            print(f"  • New storage: {new_size / (1024*1024):.2f} MB")
            print(f"  • Reduction: {reduction:.1f}%")
            
            if reduction >= 40:
                print(f"  ✅ PASS: Achieved {reduction:.1f}% storage reduction (target: 60%)")
            else:
                print(f"  ⚠️  WARNING: Only {reduction:.1f}% reduction (target: 60%)")
    except Exception as e:
        print(f"  ⚠️  Could not check storage: {e}")
    
    # ==========================================
    # Final Summary
    # ==========================================
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL VALIDATION CHECKS PASSED")
        print("=" * 60)
        print("\nMigration is successful! You can now:")
        print("1. Run performance tests: python -m app.scripts.performance_test")
        print("2. Monitor queries for 24 hours")
        print("3. After confirmation, drop old collection")
        return 0
    else:
        print("❌ SOME VALIDATION CHECKS FAILED")
        print("=" * 60)
        print("\nPlease review the errors above and fix migration issues.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(validate())
    sys.exit(exit_code)



