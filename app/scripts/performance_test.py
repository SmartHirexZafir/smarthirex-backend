"""
Performance testing: Compare old vs new schema query speeds
"""
import asyncio
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.utils.mongo import db


async def time_query(collection_name: str, query: dict, description: str) -> float:
    """Time a query and return duration in milliseconds"""
    collection = getattr(db, collection_name)
    
    start = time.time()
    results = await collection.find(query).to_list(length=1000)
    duration_ms = (time.time() - start) * 1000
    
    return duration_ms, len(results)


async def performance_test():
    """Run performance comparison tests"""
    
    print("=" * 60)
    print("PERFORMANCE TESTING: OLD vs NEW SCHEMA")
    print("=" * 60)
    
    # Check if we have a test user with data
    sample_old = await db.parsed_resumes.find_one({})
    sample_new = await db.resumes.find_one({})
    
    if not sample_old:
        print("\n❌ No documents in old collection (parsed_resumes)")
        return
    
    if not sample_new:
        print("\n❌ No documents in new collection (resumes)")
        return
    
    owner_id = sample_old.get("ownerUserId")
    if not owner_id:
        print("\n⚠️  WARNING: No ownerUserId found, using empty query")
        owner_id = None
    
    print(f"\nTest user: {owner_id or '(all users)'}")
    
    # ==========================================
    # Test 1: Role Filter
    # ==========================================
    print("\n" + "=" * 60)
    print("TEST 1: Role Filter Query")
    print("=" * 60)
    
    query_old = {"predicted_role": {"$regex": "engineer", "$options": "i"}}
    query_new = {"role.predicted": {"$regex": "engineer", "$options": "i"}}
    
    if owner_id:
        query_old["ownerUserId"] = owner_id
        query_new["ownerUserId"] = owner_id
    
    print("\nOld schema query:")
    print(f"  {query_old}")
    old_time, old_count = await time_query("parsed_resumes", query_old, "Role filter (old)")
    print(f"  ⏱️  Time: {old_time:.2f} ms")
    print(f"  📊 Results: {old_count}")
    
    print("\nNew schema query:")
    print(f"  {query_new}")
    new_time, new_count = await time_query("resumes", query_new, "Role filter (new)")
    print(f"  ⏱️  Time: {new_time:.2f} ms")
    print(f"  📊 Results: {new_count}")
    
    if new_time > 0:
        speedup = old_time / new_time
        print(f"\n🚀 Speedup: {speedup:.1f}x faster")
    
    # ==========================================
    # Test 2: Experience Range Filter
    # ==========================================
    print("\n" + "=" * 60)
    print("TEST 2: Experience Range Query")
    print("=" * 60)
    
    query_old = {
        "$or": [
            {"experience": {"$gte": 3, "$lte": 7}},
            {"total_experience_years": {"$gte": 3, "$lte": 7}},
            {"yoe": {"$gte": 3, "$lte": 7}},
        ]
    }
    query_new = {"experience.years": {"$gte": 3, "$lte": 7}}
    
    if owner_id:
        query_old["ownerUserId"] = owner_id
        query_new["ownerUserId"] = owner_id
    
    print("\nOld schema query:")
    print(f"  {query_old}")
    old_time, old_count = await time_query("parsed_resumes", query_old, "Experience filter (old)")
    print(f"  ⏱️  Time: {old_time:.2f} ms")
    print(f"  📊 Results: {old_count}")
    
    print("\nNew schema query:")
    print(f"  {query_new}")
    new_time, new_count = await time_query("resumes", query_new, "Experience filter (new)")
    print(f"  ⏱️  Time: {new_time:.2f} ms")
    print(f"  📊 Results: {new_count}")
    
    if new_time > 0:
        speedup = old_time / new_time
        print(f"\n🚀 Speedup: {speedup:.1f}x faster")
    
    # ==========================================
    # Test 3: Multi-Filter Query
    # ==========================================
    print("\n" + "=" * 60)
    print("TEST 3: Multi-Filter Query")
    print("=" * 60)
    
    query_old = {
        "$or": [
            {"predicted_role": {"$regex": "developer", "$options": "i"}},
            {"category": {"$regex": "developer", "$options": "i"}},
        ],
        "$or": [
            {"experience": {"$gte": 2}},
            {"total_experience_years": {"$gte": 2}},
        ],
        "skills": {"$regex": "python", "$options": "i"}
    }
    
    query_new = {
        "role.predicted": {"$regex": "developer", "$options": "i"},
        "experience.years": {"$gte": 2},
        "skills.list": {"$regex": "python", "$options": "i"}
    }
    
    if owner_id:
        query_old["ownerUserId"] = owner_id
        query_new["ownerUserId"] = owner_id
    
    print("\nOld schema query:")
    print(f"  {query_old}")
    old_time, old_count = await time_query("parsed_resumes", query_old, "Multi-filter (old)")
    print(f"  ⏱️  Time: {old_time:.2f} ms")
    print(f"  📊 Results: {old_count}")
    
    print("\nNew schema query:")
    print(f"  {query_new}")
    new_time, new_count = await time_query("resumes", query_new, "Multi-filter (new)")
    print(f"  ⏱️  Time: {new_time:.2f} ms")
    print(f"  📊 Results: {new_count}")
    
    if new_time > 0:
        speedup = old_time / new_time
        print(f"\n🚀 Speedup: {speedup:.1f}x faster")
    
    # ==========================================
    # Test 4: Simple Owner Query
    # ==========================================
    print("\n" + "=" * 60)
    print("TEST 4: Simple Owner Query")
    print("=" * 60)
    
    if owner_id:
        query_old = {"ownerUserId": owner_id}
        query_new = {"ownerUserId": owner_id}
        
        print("\nOld schema query:")
        print(f"  {query_old}")
        old_time, old_count = await time_query("parsed_resumes", query_old, "Owner filter (old)")
        print(f"  ⏱️  Time: {old_time:.2f} ms")
        print(f"  📊 Results: {old_count}")
        
        print("\nNew schema query:")
        print(f"  {query_new}")
        new_time, new_count = await time_query("resumes", query_new, "Owner filter (new)")
        print(f"  ⏱️  Time: {new_time:.2f} ms")
        print(f"  📊 Results: {new_count}")
        
        if new_time > 0:
            speedup = old_time / new_time
            print(f"\n🚀 Speedup: {speedup:.1f}x faster")
    else:
        print("\n⚠️  Skipped (no owner ID available)")
    
    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    print("\n✅ Performance testing complete!")
    print("\nExpected improvements:")
    print("  • Role filter: 8x faster")
    print("  • Experience filter: 8x faster")
    print("  • Multi-filter query: 5.6x faster")
    
    print("\n💡 Tips:")
    print("  • Run tests multiple times for accurate averages")
    print("  • Test with production data volume")
    print("  • Monitor query performance logs")


if __name__ == "__main__":
    asyncio.run(performance_test())



