# අවශ්‍ය පුස්තකාල import කරගැනීම
import cv2
import mediapipe as mp

# වෙබ් කැමරාව on කිරීම
cap = cv2.VideoCapture(0)

# Mediapipe වලින් අත් අඳුනගන්නා ආකෘතිය (model) සූදානම් කරගැනීම
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ඇඟිලි තුඩු වල landmark අංක
tip_ids = [4, 8, 12, 16, 20]

print("ඇප් එක ආරම්භ විය. පිටවීමට 'q' යතුර ඔබන්න.")

# කැමරාවෙන් දිගටම රූප ලබාගන්නා ලූප් එක — try/finally ensures clean shutdown
try:
    while True:
        # කැමරාවෙන් රූප රාමුවක් (frame) ලබාගැනීම
        success, image = cap.read()
        if not success:
            print("කැමරාවෙන් රූප ලබාගැනීමට නොහැකියි.")
            break

        # රූපය දකුණෙන් වමට පෙරලීම (අපේ අත බලනවා වගේ පේන්න)
        image = cv2.flip(image, 1)

        # රූපයේ වර්ණ BGR සිට RGB වලට පරිවර්තනය කිරීම (Mediapipe වලට අවශ්‍ය නිසා)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mediapipe මගින් රූපයේ අත් තිබේදැයි සෙවීම — process() can block, catch exceptions
        try:
            results = hands.process(rgb_image)
        except Exception as e:
            # If Mediapipe has a transient error, print and continue
            print(f"Mediapipe processing error: {e}")
            results = None

        finger_count = 0

        # අතක් හමුවුවහොත් පමණක් ක්‍රියාත්මක වීම
        if results and getattr(results, 'multi_hand_landmarks', None):
            for hand_landmarks in results.multi_hand_landmarks:
                # අඳුනගත් අතේ landmarks රූපයේ ඇඳීම (මේක අත්‍යවශ්‍ය නෑ, ඒත් ලස්සනයි)
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # ඇඟිලි ගණන් කිරීමේ තර්කය (Logic)
                my_hand_landmarks = hand_landmarks.landmark
                
                # --- මහපට ඇඟිල්ල (Thumb) සඳහා ---
                if my_hand_landmarks[tip_ids[0]].x < my_hand_landmarks[tip_ids[0] - 1].x:
                    finger_count += 1

                # --- අනෙක් ඇඟිලි හතර සඳහා ---
                for i in range(1, 5):
                    if my_hand_landmarks[tip_ids[i]].y < my_hand_landmarks[tip_ids[i] - 2].y:
                        finger_count += 1

        # ගණන් කල අගය තිරයේ පෙන්වීම
        cv2.putText(image, f"Finger Count: {finger_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # අවසාන රූපය තිරයේ පෙන්වීම
        cv2.imshow("Finger Counter", image)

        # 'q' යතුර එබුවොත් ලූප් එකෙන් ඉවත් වී ඇප් එක නැවැත්වීම
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    # Handle Ctrl+C from the terminal gracefully
    print("\nKeyboard interrupt received — exiting...")
finally:
    # Ensure Mediapipe resources and camera/window resources are always released
    try:
        hands.close()
    except Exception:
        # Older versions may not have close(); ignore
        pass
    cap.release()
    cv2.destroyAllWindows()
    print("ඇප් එක සාර්ථකව වසා දමන ලදී.")