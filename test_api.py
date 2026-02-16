"""
Extended Brain API Testing Script
Tests all authentication and registration endpoints
"""

import asyncio
import httpx
import json
from datetime import datetime


class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_phone = "+911234567890"
        self.test_data = {
            "name": "Test User",
            "email": f"test_{datetime.now().timestamp()}@example.com",
            "age": 25,
            "occupation": "Developer",
            "phone_number": self.test_phone,
            "password": "testpass123",
            "timezone": "Asia/Kolkata"
        }
        self.otp_code = None
    
    async def test_health(self):
        """Test health endpoint"""
        print("\n🔍 Testing health endpoint...")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    data = response.json()
                    print("✅ Health check passed")
                    print(f"   Status: {data['status']}")
                    return True
                else:
                    print(f"❌ Health check failed: {response.status_code}")
                    return False
            except Exception as e:
                print(f"❌ Health check error: {e}")
                return False
    
    async def test_send_otp(self):
        """Test OTP sending"""
        print("\n📱 Testing OTP send...")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/auth/send-otp",
                    json={"phone_number": self.test_phone}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print("✅ OTP sent successfully")
                    print(f"   Message: {data.get('message')}")
                    print(f"   Expires in: {data.get('expires_in')} minutes")
                    print("\n⚠️  Check your Railway logs or WhatsApp for the OTP code")
                    return True
                else:
                    print(f"❌ OTP send failed: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return False
            except Exception as e:
                print(f"❌ OTP send error: {e}")
                return False
    
    async def test_verify_otp(self, otp: str):
        """Test OTP verification"""
        print(f"\n🔐 Testing OTP verification with code: {otp}...")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/auth/verify-otp",
                    json={
                        "phone_number": self.test_phone,
                        "otp": otp
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('verified'):
                        print("✅ OTP verified successfully")
                        self.otp_code = otp
                        return True
                    else:
                        print(f"❌ OTP verification failed: {data.get('message')}")
                        return False
                else:
                    print(f"❌ OTP verify failed: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return False
            except Exception as e:
                print(f"❌ OTP verify error: {e}")
                return False
    
    async def test_register_user(self):
        """Test user registration"""
        print("\n👤 Testing user registration...")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/users/register",
                    json=self.test_data
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print("✅ User registered successfully")
                    print(f"   User ID: {data.get('user_id')}")
                    print(f"   Message: {data.get('message')}")
                    return True
                else:
                    print(f"❌ Registration failed: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return False
            except Exception as e:
                print(f"❌ Registration error: {e}")
                return False
    
    async def test_login(self):
        """Test user login"""
        print("\n🔑 Testing user login...")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/auth/login",
                    json={
                        "phone_number": self.test_phone,
                        "password": self.test_data["password"]
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print("✅ Login successful")
                    print(f"   User: {data.get('user', {}).get('name')}")
                    return True
                else:
                    print(f"❌ Login failed: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return False
            except Exception as e:
                print(f"❌ Login error: {e}")
                return False
    
    async def run_full_test(self):
        """Run complete test suite"""
        print("=" * 60)
        print("🧪 Extended Brain API Test Suite")
        print("=" * 60)
        print(f"Testing API at: {self.base_url}")
        print(f"Test phone: {self.test_phone}")
        
        results = {
            "health": False,
            "send_otp": False,
            "verify_otp": False,
            "register": False,
            "login": False
        }
        
        # Test 1: Health check
        results["health"] = await self.test_health()
        
        if not results["health"]:
            print("\n❌ Health check failed. Cannot continue.")
            return results
        
        # Test 2: Send OTP
        results["send_otp"] = await self.test_send_otp()
        
        if not results["send_otp"]:
            print("\n❌ OTP send failed. Cannot continue.")
            return results
        
        # Prompt for OTP
        print("\n" + "=" * 60)
        otp_input = input("Enter the OTP code you received: ")
        print("=" * 60)
        
        # Test 3: Verify OTP
        results["verify_otp"] = await self.test_verify_otp(otp_input)
        
        if not results["verify_otp"]:
            print("\n❌ OTP verification failed. Cannot continue.")
            return results
        
        # Test 4: Register user
        results["register"] = await self.test_register_user()
        
        # Test 5: Login
        if results["register"]:
            await asyncio.sleep(1)  # Brief pause
            results["login"] = await self.test_login()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Test Results Summary")
        print("=" * 60)
        
        for test, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test.upper():20} {status}")
        
        print("=" * 60)
        
        total_passed = sum(results.values())
        total_tests = len(results)
        
        if total_passed == total_tests:
            print(f"\n🎉 All tests passed! ({total_passed}/{total_tests})")
        else:
            print(f"\n⚠️  Some tests failed. ({total_passed}/{total_tests} passed)")
        
        return results


async def main():
    """Main test runner"""
    import sys
    
    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    base_url = 'https://extended-brain-backend-production.up.railway.app'
    
    print(f"\n🚀 Starting tests against: {base_url}")
    print("Make sure your API is running!\n")
    
    tester = APITester(base_url)
    await tester.run_full_test()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Extended Brain API Testing Script")
    print("=" * 60)
    print("\nUsage:")
    print("  python test_api.py                    # Test localhost")
    print("  python test_api.py https://your.app   # Test production")
    print("\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite failed: {e}")
