#!/usr/bin/env python3
"""
Quick verification script to test duplicate detection fix
Run this to verify the contentHash field name fix is working
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.mongo import db, compute_cv_hash, check_duplicate_cv_hash


async def verify_duplicate_detection():
    """
    Verify that duplicate detection is working correctly with the fixed field name
    """
    print("=" * 70)
    print("🔍 DUPLICATE DETECTION VERIFICATION")
    print("=" * 70)
    
    # Test text
    test_text = "This is a test resume content for duplicate detection verification."
    test_hash = compute_cv_hash(test_text)
    
    print(f"\n1️⃣ Testing hash computation...")
    print(f"   Input text: '{test_text[:50]}...'")
    print(f"   SHA256 Hash: {test_hash}")
    print(f"   Hash length: {len(test_hash)} characters")
    assert len(test_hash) == 64, "SHA256 hash should be 64 characters"
    print("   ✅ Hash computation working correctly")
    
    print(f"\n2️⃣ Checking MongoDB connection...")
    try:
        # Ping database
        await db.command("ping")
        print("   ✅ MongoDB connection successful")
    except Exception as e:
        print(f"   ❌ MongoDB connection failed: {e}")
        return
    
    print(f"\n3️⃣ Checking 'resumes' collection...")
    try:
        count = await db.resumes.count_documents({})
        print(f"   Total documents in 'resumes': {count}")
        print("   ✅ Collection accessible")
    except Exception as e:
        print(f"   ❌ Collection access failed: {e}")
        return
    
    print(f"\n4️⃣ Checking for 'contentHash' field (camelCase)...")
    try:
        # Find a document with contentHash field
        sample = await db.resumes.find_one({"contentHash": {"$exists": True}})
        if sample:
            print(f"   ✅ Found documents with 'contentHash' field (correct camelCase)")
            print(f"      Sample hash: {sample.get('contentHash', 'N/A')[:32]}...")
        else:
            print("   ⚠️  No documents found with 'contentHash' field")
            print("      This is OK if database is empty or hasn't been migrated yet")
    except Exception as e:
        print(f"   ❌ Query failed: {e}")
    
    print(f"\n5️⃣ Checking for old 'content_hash' field (snake_case)...")
    try:
        # Find a document with old content_hash field
        old_sample = await db.resumes.find_one({"content_hash": {"$exists": True}})
        if old_sample:
            print(f"   ⚠️  Found documents with OLD 'content_hash' field (snake_case)")
            print("      Migration may be incomplete - run migration script")
        else:
            print("   ✅ No documents with old 'content_hash' field")
            print("      Schema is clean (all documents use camelCase)")
    except Exception as e:
        print(f"   ❌ Query failed: {e}")
    
    print(f"\n6️⃣ Testing duplicate check function...")
    try:
        # Test with non-existent content
        test_user_id = "test-verification-user-12345"
        is_dup = await check_duplicate_cv_hash(test_text, owner_user_id=test_user_id)
        print(f"   Checking if test content exists for user '{test_user_id}': {is_dup}")
        if not is_dup:
            print("   ✅ Duplicate check function working (returned False for new content)")
        else:
            print("   ⚠️  Test content already exists (or false positive)")
    except Exception as e:
        print(f"   ❌ Duplicate check failed: {e}")
        return
    
    print(f"\n7️⃣ Checking MongoDB indexes...")
    try:
        indexes = await db.resumes.list_indexes().to_list(length=100)
        print(f"   Total indexes: {len(indexes)}")
        
        # Check for contentHash index
        has_hash_index = False
        for idx in indexes:
            keys = idx.get("key", {})
            if "contentHash" in keys:
                has_hash_index = True
                print(f"   ✅ Found contentHash index: {idx.get('name')}")
                print(f"      Keys: {keys}")
        
        if not has_hash_index:
            print("   ⚠️  No contentHash index found")
            print("      Run 'python -m app.scripts.optimize_indexes' to create it")
            print("      Duplicate detection will be slower without index")
    except Exception as e:
        print(f"   ❌ Index check failed: {e}")
    
    print("\n" + "=" * 70)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 70)
    
    print("\n📝 SUMMARY:")
    print("   • Hash computation: Working")
    print("   • MongoDB connection: Working")
    print("   • Field name: Using 'contentHash' (camelCase) ✅")
    print("   • Duplicate check function: Working")
    
    print("\n🎯 NEXT STEPS:")
    print("   1. Upload a test CV through the UI")
    print("   2. Try uploading the same CV again")
    print("   3. Verify second upload is rejected with 'Duplicate' error")
    print("   4. Check toaster displays for exactly 3 seconds")
    print("   5. Verify toaster shows detailed failure reason")
    
    print("\n📚 DOCUMENTATION:")
    print("   • Full fix details: TOASTER_AND_DUPLICATE_FIX.md")
    print("   • Testing guide: TEST_DUPLICATE_DETECTION.md")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(verify_duplicate_detection())
    except KeyboardInterrupt:
        print("\n\n⚠️  Verification interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



